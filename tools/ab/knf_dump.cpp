// knf_dump: dump REAL kaldi-native-fbank features for a PCM16 mono 16 kHz WAV.
//
// Used to A/B the Fbank configuration that feeds NeMo Parakeet TDT:
//   --profile testonnx  : what sherpa-onnx/scripts/nemo/parakeet-tdt-0.6b-v2/test_onnx.py sets
//                         (hann, low_freq 0, high_freq = Nyquist, is_librosa, no DC, snip_edges=true,
//                          log(max(x, eps)), +2 s zero tail)
//   --profile sherpa    : what the sherpa-onnx C++ offline recognizer (OfflineRecognizerTransducerNeMoImpl)
//                         sets (povey window, low_freq 0, high_freq -400, is_librosa, no DC,
//                          snip_edges=false, log(x + 2^-24), no extra tail)
// Every knob can be overridden individually.
//
// Output: <out>.raw.f32  float32 row-major [T,128], BEFORE normalisation
//         <out>.norm.f32 float32 row-major [T,128], per_feature normalised (unbiased std + 1e-5)
// stdout: one JSON line with sizes and statistics.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

static bool ReadWav(const std::string &path, std::vector<float> *out, int *sr) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::vector<char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (d.size() < 44 || std::memcmp(d.data(), "RIFF", 4) || std::memcmp(d.data() + 8, "WAVE", 4)) return false;
  size_t pos = 12;
  int ch = 0, bits = 0, fmt = 0;
  while (pos + 8 <= d.size()) {
    uint32_t sz;
    std::memcpy(&sz, d.data() + pos + 4, 4);
    std::string id(d.data() + pos, 4);
    if (id == "fmt ") {
      uint16_t a, c, b;
      uint32_t r;
      std::memcpy(&a, d.data() + pos + 8, 2);
      std::memcpy(&c, d.data() + pos + 10, 2);
      std::memcpy(&r, d.data() + pos + 12, 4);
      std::memcpy(&b, d.data() + pos + 22, 2);
      fmt = a; ch = c; *sr = (int)r; bits = b;
    } else if (id == "data") {
      if (fmt != 1 || bits != 16 || ch < 1) return false;
      size_t n = std::min<size_t>(sz, d.size() - pos - 8) / 2 / ch;
      out->resize(n);
      const int16_t *p = reinterpret_cast<const int16_t *>(d.data() + pos + 8);
      for (size_t i = 0; i < n; ++i) (*out)[i] = p[i * ch] / 32768.0f;
      return true;
    }
    pos += 8 + sz + (sz & 1);
  }
  return false;
}

int main(int argc, char **argv) {
  std::string wav, out = "features", profile = "testonnx";
  std::string window, log_mode;
  float pad_s = -1, high = std::numeric_limits<float>::quiet_NaN();
  int snip = -1;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--wav") wav = next();
    else if (a == "--out") out = next();
    else if (a == "--profile") profile = next();
    else if (a == "--window") window = next();
    else if (a == "--high-freq") high = std::stof(next());
    else if (a == "--snip-edges") snip = std::stoi(next());
    else if (a == "--log") log_mode = next();  // floor | additive
    else if (a == "--pad-seconds") pad_s = std::stof(next());
    else { std::cerr << "unknown arg " << a << "\n"; return 2; }
  }
  if (wav.empty()) { std::cerr << "usage: knf_dump --wav f.wav [--profile testonnx|sherpa] [--out prefix] [--window hann|povey] [--high-freq hz] [--snip-edges 0|1] [--log floor|additive] [--pad-seconds s]\n"; return 2; }

  bool sherpa = profile == "sherpa";
  if (window.empty()) window = sherpa ? "povey" : "hann";
  if (std::isnan(high)) high = sherpa ? -400.0f : 0.0f;
  if (snip < 0) snip = sherpa ? 0 : 1;
  if (log_mode.empty()) log_mode = sherpa ? "additive" : "floor";
  if (pad_s < 0) pad_s = sherpa ? 0.0f : 2.0f;

  std::vector<float> audio;
  int sr = 0;
  if (!ReadWav(wav, &audio, &sr) || sr != 16000) { std::cerr << "need PCM16 16k WAV\n"; return 1; }
  size_t n_audio = audio.size();
  audio.resize(audio.size() + (size_t)(16000 * pad_s), 0.0f);

  knf::FbankOptions o;
  o.frame_opts.dither = 0;
  o.frame_opts.remove_dc_offset = false;
  o.frame_opts.window_type = window;
  o.frame_opts.snip_edges = snip != 0;
  o.mel_opts.low_freq = 0;
  o.mel_opts.high_freq = high;
  o.mel_opts.num_bins = 128;
  o.mel_opts.is_librosa = true;
  if (log_mode == "additive") {
    o.use_log_floor = false;
    o.eps = std::numeric_limits<float>::epsilon() / 2;
  }
  knf::OnlineFbank fbank(o);
  fbank.AcceptWaveform(16000, audio.data(), (int32_t)audio.size());
  fbank.InputFinished();
  int T = fbank.NumFramesReady();
  const int D = 128;
  std::vector<float> raw((size_t)T * D);
  for (int t = 0; t < T; ++t) std::memcpy(&raw[(size_t)t * D], fbank.GetFrame(t), D * sizeof(float));

  std::vector<float> norm(raw.size());
  double mean_of_means = 0, mean_of_std = 0;
  for (int c = 0; c < D; ++c) {
    double m = 0;
    for (int t = 0; t < T; ++t) m += raw[(size_t)t * D + c];
    m /= T;
    double v = 0;
    for (int t = 0; t < T; ++t) { double x = raw[(size_t)t * D + c] - m; v += x * x; }
    v /= (T > 1 ? T - 1 : 1);
    double s = std::sqrt(v) + 1e-5;
    mean_of_means += m; mean_of_std += std::sqrt(v);
    for (int t = 0; t < T; ++t) norm[(size_t)t * D + c] = (float)((raw[(size_t)t * D + c] - m) / s);
  }
  float mn = 1e30f, mx = -1e30f;
  for (float x : raw) { mn = std::min(mn, x); mx = std::max(mx, x); }
  std::ofstream(out + ".raw.f32", std::ios::binary).write((const char *)raw.data(), raw.size() * 4);
  std::ofstream(out + ".norm.f32", std::ios::binary).write((const char *)norm.data(), norm.size() * 4);
  std::printf("{\"profile\":\"%s\",\"window\":\"%s\",\"high_freq\":%g,\"snip_edges\":%d,\"log\":\"%s\",\"pad_seconds\":%g,"
              "\"audio_samples\":%zu,\"padded_samples\":%zu,\"frames\":%d,\"dim\":%d,\"raw_min\":%.4f,\"raw_max\":%.4f,"
              "\"raw_mean_of_bin_means\":%.4f,\"raw_mean_of_bin_std\":%.4f}\n",
              profile.c_str(), window.c_str(), high, snip, log_mode.c_str(), pad_s, n_audio, audio.size(), T, D, mn, mx,
              mean_of_means / D, mean_of_std / D);
  return 0;
}
