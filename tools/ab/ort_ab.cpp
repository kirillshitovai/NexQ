// ort_ab: reference runner for NeMo Parakeet TDT 0.6B v3 (sherpa-onnx int8 export) on ONNX Runtime.
//
//   ort_ab info   --model-dir D                         print input/output signatures of the 3 models + metadata
//   ort_ab decode --model-dir D --features F.f32        decode [T,128] float32 features (row-major, raw)
//                 [--norm per_feature|per_feature_biased|none] [--tdt sherpa|nemo] [--threads N] [--opt 0|1|2|99] [--dump-tokens]
//   ort_ab serve  --model-dir D                         binary request/response server on stdin/stdout, used
//                                                       by the Rust test backend (see nemo_parakeet_core.rs)
//
// The decode loop is a line-by-line port of
//   sherpa-onnx/scripts/nemo/parakeet-tdt-0.6b-v2/test_onnx.py   (--tdt sherpa)
// and, for comparison only, of NeMo GreedyTDTInfer's frame handling (--tdt nemo).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

static int g_opt = -1;  // --opt 0|1|2|99 : graph optimisation level (default: ORT default = ALL)

struct Net {
  Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "ort_ab"};
  Ort::SessionOptions so;
  std::unique_ptr<Ort::Session> enc, dec, joi;
  Ort::AllocatorWithDefaultOptions alloc;
  Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<std::string> enc_in, enc_out, dec_in, dec_out, joi_in, joi_out;
  std::string normalize_type;
  int pred_layers = 2, pred_hidden = 640;
  std::vector<std::string> tokens;  // id -> token
  int vocab = 0, blank = 0;

  static std::vector<std::string> Names(Ort::Session &s, bool in) {
    std::vector<std::string> v;
    size_t n = in ? s.GetInputCount() : s.GetOutputCount();
    Ort::AllocatorWithDefaultOptions a;
    for (size_t i = 0; i < n; ++i)
      v.push_back(in ? s.GetInputNameAllocated(i, a).get() : s.GetOutputNameAllocated(i, a).get());
    return v;
  }

  void Load(const std::string &dir, int threads) {
    so.SetIntraOpNumThreads(threads);
    so.SetInterOpNumThreads(1);
    if (g_opt == 0) so.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    else if (g_opt == 1) so.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    else if (g_opt == 2) so.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    else if (g_opt == 99) so.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    auto pick = [&](const char *stem) {
      std::string a = dir + "/" + stem + ".int8.onnx";
      std::ifstream f(a);
      return f.good() ? a : dir + "/" + stem + ".onnx";
    };
    enc = std::make_unique<Ort::Session>(env, pick("encoder").c_str(), so);
    dec = std::make_unique<Ort::Session>(env, pick("decoder").c_str(), so);
    joi = std::make_unique<Ort::Session>(env, pick("joiner").c_str(), so);
    enc_in = Names(*enc, true); enc_out = Names(*enc, false);
    dec_in = Names(*dec, true); dec_out = Names(*dec, false);
    joi_in = Names(*joi, true); joi_out = Names(*joi, false);
    auto md = enc->GetModelMetadata();
    auto get = [&](const char *k) -> std::string {
      auto v = md.LookupCustomMetadataMapAllocated(k, alloc);
      return v ? std::string(v.get()) : std::string();
    };
    normalize_type = get("normalize_type");
    if (!get("pred_rnn_layers").empty()) pred_layers = std::stoi(get("pred_rnn_layers"));
    if (!get("pred_hidden").empty()) pred_hidden = std::stoi(get("pred_hidden"));
    std::ifstream tf(dir + "/tokens.txt");
    std::string line;
    std::map<int, std::string> m;
    while (std::getline(tf, line)) {
      if (line.empty()) continue;
      auto p = line.rfind(' ');
      m[std::stoi(line.substr(p + 1))] = line.substr(0, p);
    }
    vocab = (int)m.size();
    blank = vocab - 1;
    tokens.resize(vocab);
    for (auto &kv : m) tokens[kv.first] = kv.second;
  }

  // encoder: feats channel-major [128,T]
  std::vector<float> Encode(const std::vector<float> &x_ct, int T, int *t_out, int *enc_len) {
    std::vector<int64_t> shape{1, 128, T};
    int64_t len = T;
    std::vector<int64_t> lshape{1};
    Ort::Value in[2] = {Ort::Value::CreateTensor<float>(mi, const_cast<float *>(x_ct.data()), x_ct.size(), shape.data(), 3),
                        Ort::Value::CreateTensor<int64_t>(mi, &len, 1, lshape.data(), 1)};
    std::vector<const char *> inn, outn;
    for (auto &s : enc_in) inn.push_back(s.c_str());
    for (size_t i = 0; i < 2 && i < enc_out.size(); ++i) outn.push_back(enc_out[i].c_str());
    auto out = enc->Run(Ort::RunOptions{nullptr}, inn.data(), in, 2, outn.data(), outn.size());
    auto ts = out[0].GetTensorTypeAndShapeInfo();
    auto sh = ts.GetShape();
    if (sh.size() != 3 || sh[0] != 1 || sh[1] != 1024) {
      std::fprintf(stderr, "unexpected encoder output shape [%lld,%lld,%lld]\n", (long long)sh[0], (long long)sh[1], (long long)sh[2]);
      std::exit(3);
    }
    *t_out = (int)sh[2];
    *enc_len = out.size() > 1 ? (int)out[1].GetTensorData<int64_t>()[0] : -1;
    const float *p = out[0].GetTensorData<float>();
    return std::vector<float>(p, p + (size_t)1024 * sh[2]);
  }

  // decoder step. token int32 [1,1], length int32 [1], two states [L,1,H]
  void Decode(int token, const std::vector<float> &s0, const std::vector<float> &s1, std::vector<float> *dec_out,
              std::vector<float> *n0, std::vector<float> *n1) {
    int32_t tok = token, one = 1;
    std::vector<int64_t> tshape{1, 1}, lshape{1}, sshape{pred_layers, 1, pred_hidden};
    Ort::Value in[4] = {
        Ort::Value::CreateTensor<int32_t>(mi, &tok, 1, tshape.data(), 2),
        Ort::Value::CreateTensor<int32_t>(mi, &one, 1, lshape.data(), 1),
        Ort::Value::CreateTensor<float>(mi, const_cast<float *>(s0.data()), s0.size(), sshape.data(), 3),
        Ort::Value::CreateTensor<float>(mi, const_cast<float *>(s1.data()), s1.size(), sshape.data(), 3)};
    std::vector<const char *> inn, outn;
    for (auto &s : dec_in) inn.push_back(s.c_str());
    for (size_t i = 0; i < 4; ++i) outn.push_back(dec_out_name(i));
    auto out = dec->Run(Ort::RunOptions{nullptr}, inn.data(), in, 4, outn.data(), 4);
    auto cp = [&](Ort::Value &v, std::vector<float> *o) {
      size_t n = v.GetTensorTypeAndShapeInfo().GetElementCount();
      const float *p = v.GetTensorData<float>();
      o->assign(p, p + n);
    };
    cp(out[0], dec_out); cp(out[2], n0); cp(out[3], n1);
  }
  const char *dec_out_name(size_t i) { return dec_out[i].c_str(); }

  std::vector<float> Join(const float *enc_frame, const float *dec_frame) {
    std::vector<int64_t> es{1, 1024, 1}, ds{1, pred_hidden, 1};
    Ort::Value in[2] = {Ort::Value::CreateTensor<float>(mi, const_cast<float *>(enc_frame), 1024, es.data(), 3),
                        Ort::Value::CreateTensor<float>(mi, const_cast<float *>(dec_frame), pred_hidden, ds.data(), 3)};
    std::vector<const char *> inn{joi_in[0].c_str(), joi_in[1].c_str()}, outn{joi_out[0].c_str()};
    auto out = joi->Run(Ort::RunOptions{nullptr}, inn.data(), in, 2, outn.data(), 1);
    size_t n = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
    const float *p = out[0].GetTensorData<float>();
    return std::vector<float>(p, p + n);
  }
};

static const char *TypeName(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "f32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "i32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "i64";
    default: return "other";
  }
}

static void PrintIO(const char *label, Ort::Session &s) {
  Ort::AllocatorWithDefaultOptions a;
  for (int in = 1; in >= 0; --in) {
    size_t n = in ? s.GetInputCount() : s.GetOutputCount();
    for (size_t i = 0; i < n; ++i) {
      auto name = in ? s.GetInputNameAllocated(i, a) : s.GetOutputNameAllocated(i, a);
      auto ti = in ? s.GetInputTypeInfo(i) : s.GetOutputTypeInfo(i);
      auto info = ti.GetTensorTypeAndShapeInfo();
      std::printf("  %-8s %-6s %-22s %s [", label, in ? "input" : "output", name.get(), TypeName(info.GetElementType()));
      auto sh = info.GetShape();
      for (size_t k = 0; k < sh.size(); ++k) std::printf("%s%lld", k ? "," : "", (long long)sh[k]);
      std::printf("]\n");
    }
  }
}

static std::vector<float> ReadF32(const std::string &p) {
  std::ifstream f(p, std::ios::binary);
  std::vector<char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  std::vector<float> v(d.size() / 4);
  std::memcpy(v.data(), d.data(), v.size() * 4);
  return v;
}

static std::string Detok(const Net &n, const std::vector<int> &ids) {
  std::string s;
  for (int i : ids) s += n.tokens[i];
  std::string out;
  for (size_t i = 0; i < s.size();) {  // U+2581 = E2 96 81
    if ((unsigned char)s[i] == 0xE2 && i + 2 < s.size() && (unsigned char)s[i + 1] == 0x96 && (unsigned char)s[i + 2] == 0x81) {
      out += ' ';
      i += 3;
    } else out += s[i++];
  }
  size_t a = out.find_first_not_of(' '), b = out.find_last_not_of(' ');
  return a == std::string::npos ? "" : out.substr(a, b - a + 1);
}

static std::string JsonEsc(const std::string &s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if ((unsigned char)c < 0x20) o += ' ';
    else o += c;
  }
  return o;
}

static int ArgmaxRange(const std::vector<float> &v, size_t a, size_t b) {
  return (int)(std::max_element(v.begin() + a, v.begin() + b) - (v.begin() + a));
}

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  std::string cmd = argv[1], dir, feat, norm = "per_feature", tdt = "sherpa";
  int threads = 2, max_symbols = 10, repeats = 1;
  float perturb = 0.0f;
  unsigned seed = 1;
  bool dump_tokens = false;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--model-dir") dir = next();
    else if (a == "--features") feat = next();
    else if (a == "--norm") norm = next();
    else if (a == "--tdt") tdt = next();
    else if (a == "--threads") threads = std::stoi(next());
    else if (a == "--opt") g_opt = std::stoi(next());
    else if (a == "--max-symbols") max_symbols = std::stoi(next());
    else if (a == "--dump-tokens") dump_tokens = true;
    else if (a == "--perturb") perturb = std::stof(next());   // add N(0, sigma^2) to the NORMALISED features
    else if (a == "--repeats") repeats = std::stoi(next());   // with --perturb: independent noise draws per feature file
    else if (a == "--seed") seed = (unsigned)std::stoul(next());
  }
  Net net;
  net.Load(dir, threads);

  if (cmd == "info") {
    PrintIO("encoder", *net.enc);
    PrintIO("decoder", *net.dec);
    PrintIO("joiner", *net.joi);
    std::printf("metadata: normalize_type=%s pred_rnn_layers=%d pred_hidden=%d\n", net.normalize_type.c_str(), net.pred_layers, net.pred_hidden);
    std::printf("tokens: vocab=%d blank_id=%d blank_token=%s\n", net.vocab, net.blank, net.tokens[net.blank].c_str());
    return 0;
  }

  if (cmd == "serve") {
    auto rd = [](void *p, size_t n) { return std::fread(p, 1, n, stdin) == n; };
    auto wr = [](const void *p, size_t n) { std::fwrite(p, 1, n, stdout); };
    std::printf("%s %d %d %d %d\n", net.normalize_type.empty() ? "-" : net.normalize_type.c_str(), net.pred_layers, net.pred_hidden, net.vocab, net.blank);
    std::fflush(stdout);
    for (;;) {
      uint8_t c;
      if (!rd(&c, 1) || c == 'Q') return 0;
      if (c == 'E') {
        uint32_t T;
        rd(&T, 4);
        std::vector<float> x((size_t)128 * T);
        rd(x.data(), x.size() * 4);
        int t_out, el;
        auto e = net.Encode(x, (int)T, &t_out, &el);
        uint32_t t32 = t_out;
        wr(&t32, 4);
        wr(e.data(), e.size() * 4);
      } else if (c == 'D') {
        int32_t tok;
        rd(&tok, 4);
        size_t sn = (size_t)net.pred_layers * net.pred_hidden;
        std::vector<float> s0(sn), s1(sn), d, n0, n1;
        rd(s0.data(), sn * 4);
        rd(s1.data(), sn * 4);
        net.Decode(tok, s0, s1, &d, &n0, &n1);
        wr(d.data(), d.size() * 4);
        wr(n0.data(), n0.size() * 4);
        wr(n1.data(), n1.size() * 4);
      } else if (c == 'J') {
        std::vector<float> e(1024), d(net.pred_hidden);
        rd(e.data(), 1024 * 4);
        rd(d.data(), net.pred_hidden * 4);
        auto l = net.Join(e.data(), d.data());
        uint32_t n = (uint32_t)l.size();
        wr(&n, 4);
        wr(l.data(), l.size() * 4);
      }
      std::fflush(stdout);
    }
  }

  // ---- decode ----
  std::vector<std::string> feat_files;
  { size_t a = 0; for (;;) { size_t b = feat.find(',', a); feat_files.push_back(feat.substr(a, b == std::string::npos ? b : b - a)); if (b == std::string::npos) break; a = b + 1; } }
  for (const auto &ff : feat_files) {
  auto raw0 = ReadF32(ff);
  for (int rep = 0; rep < repeats; ++rep) {
  auto raw = raw0;
  int T = (int)(raw.size() / 128);
  if (norm == "per_feature" || norm == "per_feature_biased") {
    const bool biased = (norm == "per_feature_biased");
    for (int c = 0; c < 128; ++c) {
      double m = 0;
      for (int t = 0; t < T; ++t) m += raw[(size_t)t * 128 + c];
      m /= T;
      double v = 0;
      for (int t = 0; t < T; ++t) { double x = raw[(size_t)t * 128 + c] - m; v += x * x; }
      v /= biased ? T : (T > 1 ? T - 1 : 1);
      double s = std::sqrt(v) + 1e-5;
      for (int t = 0; t < T; ++t) raw[(size_t)t * 128 + c] = (float)((raw[(size_t)t * 128 + c] - m) / s);
    }
  }
  if (perturb > 0.0f) {
    std::mt19937 gen(seed + 1000u * rep);
    std::normal_distribution<float> nd(0.0f, perturb);
    for (auto &x : raw) x += nd(gen);
  }
  std::vector<float> ct((size_t)128 * T);
  for (int t = 0; t < T; ++t)
    for (int c = 0; c < 128; ++c) ct[(size_t)c * T + t] = raw[(size_t)t * 128 + c];
  int Te, elen;
  auto enc = net.Encode(ct, T, &Te, &elen);

  size_t sn = (size_t)net.pred_layers * net.pred_hidden;
  std::vector<float> s0(sn, 0.f), s1(sn, 0.f), dec_out, n0, n1;
  std::vector<int> ids;
  net.Decode(net.blank, s0, s1, &dec_out, &n0, &n1);
  std::vector<float> frame(1024);
  int t = 0, emitted_here = 0, joiner_calls = 0, zero_skip = 0, nonblank_zero_skip = 0;
  std::vector<int> frames_of_tokens;
  while (t < Te) {
    for (int c = 0; c < 1024; ++c) frame[c] = enc[(size_t)c * Te + t];
    auto logits = net.Join(frame.data(), dec_out.data());
    ++joiner_calls;
    if ((int)logits.size() != net.vocab + 5) { std::fprintf(stderr, "logits=%zu vocab=%d\n", logits.size(), net.vocab); return 4; }
    int tok = ArgmaxRange(logits, 0, net.vocab);
    int skip = ArgmaxRange(logits, net.vocab, logits.size());
    if (skip == 0) { ++zero_skip; if (tok != net.blank) ++nonblank_zero_skip; }
    if (tdt == "sherpa") {
      if (skip == 0) skip = 1;
      if (tok != net.blank) {
        ids.push_back(tok);
        frames_of_tokens.push_back(t);
        s0 = n0; s1 = n1;
        net.Decode(tok, s0, s1, &dec_out, &n0, &n1);
      }
      t += skip;
    } else {  // NeMo GreedyTDTInfer style
      if (tok == net.blank) skip = std::max(skip, 1);
      else {
        ids.push_back(tok);
        frames_of_tokens.push_back(t);
        s0 = n0; s1 = n1;
        net.Decode(tok, s0, s1, &dec_out, &n0, &n1);
        ++emitted_here;
        if (skip == 0 && emitted_here >= max_symbols) skip = 1;
      }
      if (skip > 0) { emitted_here = 0; t += skip; }
    }
  }
  std::string text = Detok(net, ids);
  std::printf("{\"rep\":%d,\"norm\":\"%s\",\"tdt\":\"%s\",\"feature_frames\":%d,\"encoder_frames\":%d,\"encoded_len\":%d,\"joiner_calls\":%d,"
              "\"zero_skip\":%d,\"nonblank_zero_skip\":%d,\"num_tokens\":%zu,\"text\":\"%s\"",
              rep, norm.c_str(), tdt.c_str(), T, Te, elen, joiner_calls, zero_skip, nonblank_zero_skip, ids.size(), JsonEsc(text).c_str());
  if (dump_tokens) {
    std::printf(",\"token_ids\":[");
    for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? "," : "", ids[i]);
    std::printf("]");
  }
  std::printf("}\n");
  }
  }
  return 0;
}
