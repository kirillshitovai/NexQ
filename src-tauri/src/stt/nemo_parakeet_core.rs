//! NeMo Parakeet TDT 0.6B v3 — backend independent logic (revision 3).
//!
//! This file deliberately depends on `std` only (no `ort`, `ndarray`, `rustfft`, `tokio`, `tauri`).
//! Everything that decides *what text comes out* lives here so that it can be compiled and tested
//! without the application and without ONNX Runtime:
//!
//!   * [`Fbank`]       — 128-bin log-mel front end that reproduces `kaldi-native-fbank` with the options
//!                       sherpa-onnx uses for NeMo transducers (`is_librosa = true`, Slaney mel scale
//!                       with area normalisation, `dither = 0`, `remove_dc_offset = false`).
//!   * [`normalize_per_feature`] — NeMo `per_feature` normalisation (unbiased std + 1e-5).
//!   * [`Vocab`]       — strict `tokens.txt` parsing / validation (hard errors, never warnings).
//!   * [`transcribe`]  — encoder → TDT greedy loop → text, generic over a [`Backend`] trait so the very
//!                       same loop runs against ONNX Runtime in the app and against a reference process
//!                       in the tests.
//!   * [`Segmenter`]   — RMS based phrase segmenter (NOT a VAD) with explicit events, so nothing is
//!                       dropped silently.
//!   * [`AudioMeter`]  — counters used for the diagnostic log (sample count, rate, peak, RMS).
//!
//! Numerical facts below were measured, not assumed: see `tools/ab/` and the README of this patch.
#![allow(dead_code)]

use std::collections::VecDeque;

pub const SAMPLE_RATE: usize = 16_000;
pub const FEAT_DIM: usize = 128;
pub const ENC_DIM: usize = 1024;
pub const WIN_LEN: usize = 400; // 25 ms
pub const WIN_SHIFT: usize = 160; // 10 ms
pub const FFT_LEN: usize = 512; // next power of two >= WIN_LEN (knf round_to_power_of_two)
pub const N_BINS: usize = FFT_LEN / 2 + 1;
/// Parakeet TDT 0.6B v3 predicts 5 durations (0..=4 frames) after the token logits.
pub const N_DURATIONS: usize = 5;
const NORM_EPS: f32 = 1e-5;

// ───────────────────────────────────────────────────────────── Fbank ─────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowKind {
    /// `0.5 - 0.5 cos(2πi/N)` (periodic, as in `torch.hann_window`), N = 400.
    Hann,
    /// `(0.5 - 0.5 cos(2πi/(N-1)))^0.85` — knf's default window.
    Povey,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogGuard {
    /// `ln(max(x, FLT_EPSILON))` — knf default (and what the sherpa-onnx v1.13.8 release does).
    Floor,
    /// `ln(x + 2^-24)` — NeMo's own guard, selected by newer sherpa-onnx sources.
    Additive,
}

/// Fbank configuration. Presets mirror the configurations that were compared against the real
/// `kaldi-native-fbank` and the real model (see README, "A/B results").
#[derive(Clone, Debug, PartialEq)]
pub struct FbankConfig {
    pub name: &'static str,
    pub window: WindowKind,
    /// knf semantics: `> 0` is an absolute frequency, `<= 0` is an offset from Nyquist (0 → 8000 Hz).
    pub high_freq: f32,
    pub log_guard: LogGuard,
    /// knf `snip_edges`: `true` → frames fully inside the signal; `false` → centred frames with edge reflection.
    pub snip_edges: bool,
    /// Zero samples appended before feature extraction.
    pub tail_pad_samples: usize,
}

impl FbankConfig {
    /// sherpa-onnx v1.13.8 release behaviour (no tail padding). Reproduces the release's text exactly on the test WAV.
    /// Measured (README): best on ONNX Runtime >= 1.25 (CER 1.0-1.1 %, all short fragments transcribed) but fragile on
    /// the 1.24.x build NexQ pins (CER 6.3 %, most short fragments empty). DEFAULT except on that build.
    pub fn release() -> Self {
        Self { name: "release", window: WindowKind::Hann, high_freq: 0.0, log_guard: LogGuard::Floor, snip_edges: false, tail_pad_samples: 0 }
    }
    /// The recipe of `sherpa-onnx/scripts/nemo/parakeet-tdt-0.6b-v2/test_onnx.py`: snip_edges=true and a 2 s zero tail.
    /// Measured (README): the least sensitive to the ORT version (CER 1.3 % on 1.24.2, 1.25 and 1.28); automatically
    /// selected on the known 1.24.x builds, where it is the only usable recipe.
    pub fn test_onnx() -> Self {
        Self { name: "test_onnx", window: WindowKind::Hann, high_freq: 0.0, log_guard: LogGuard::Floor, snip_edges: true, tail_pad_samples: 2 * SAMPLE_RATE }
    }
    /// sherpa-onnx master source: Povey window, 0..7600 Hz, additive log guard, snip_edges=false, no tail.
    pub fn master() -> Self {
        Self { name: "master", window: WindowKind::Povey, high_freq: -400.0, log_guard: LogGuard::Additive, snip_edges: false, tail_pad_samples: 0 }
    }
    /// test_onnx.py without its 2 s tail (isolates the effect of padding).
    pub fn hann_nopad() -> Self {
        Self { name: "hann_nopad", window: WindowKind::Hann, high_freq: 0.0, log_guard: LogGuard::Floor, snip_edges: true, tail_pad_samples: 0 }
    }
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "release" | "" => Some(Self::release()),
            "test_onnx" | "testonnx" => Some(Self::test_onnx()),
            "master" => Some(Self::master()),
            "hann_nopad" => Some(Self::hann_nopad()),
            _ => None,
        }
    }
    pub fn describe(&self) -> String {
        format!(
            "fbank={} (window={:?}, mel=128 slaney/area-norm 0..{} Hz, log={:?}, snip_edges={}, tail_pad={} samples, preemph=0.97, dither=0, dc_removal=off)",
            self.name,
            self.window,
            if self.high_freq > 0.0 { self.high_freq } else { 8000.0 + self.high_freq },
            self.log_guard,
            self.snip_edges,
            self.tail_pad_samples
        )
    }
    /// Number of feature frames for `n_samples` input samples (tail padding included).
    pub fn num_frames(&self, n_samples: usize) -> usize {
        let n = n_samples + self.tail_pad_samples;
        if self.snip_edges {
            if n < WIN_LEN { 0 } else { 1 + (n - WIN_LEN) / WIN_SHIFT }
        } else {
            (n + WIN_SHIFT / 2) / WIN_SHIFT
        }
    }
}

struct Fft {
    rev: Vec<usize>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Fft {
    fn new(n: usize) -> Self {
        assert!(n.is_power_of_two());
        let bits = n.trailing_zeros();
        let rev = (0..n).map(|i| i.reverse_bits() >> (usize::BITS - bits)).collect();
        let cos = (0..n / 2).map(|k| (2.0 * std::f64::consts::PI * k as f64 / n as f64).cos() as f32).collect();
        let sin = (0..n / 2).map(|k| (-2.0 * std::f64::consts::PI * k as f64 / n as f64).sin() as f32).collect();
        Self { rev, cos, sin }
    }
    /// In-place forward complex FFT (decimation in time).
    fn forward(&self, re: &mut [f32], im: &mut [f32]) {
        let n = re.len();
        for i in 0..n {
            let j = self.rev[i];
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let (wr, wi) = (self.cos[k * step], self.sin[k * step]);
                    let (a, b) = (start + k, start + k + half);
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            len <<= 1;
        }
    }
}

fn mel_scale_slaney(freq: f32) -> f32 {
    if freq <= 1000.0 { freq * 3.0 / 200.0 } else { 15.0 + 14.545078505785561_f32 * (freq / 1000.0).ln() }
}

fn inv_mel_scale_slaney(mel: f32) -> f32 {
    if mel <= 15.0 { 200.0f32 / 3.0 * mel } else { 1000.0 * ((mel - 15.0) * 0.06875177742094911_f32).exp() }
}

pub struct Fbank {
    cfg: FbankConfig,
    window: Vec<f32>,
    /// per mel bin: (first FFT bin, weights)
    mel: Vec<(usize, Vec<f32>)>,
    fft: Fft,
}

impl Fbank {
    pub fn new(cfg: FbankConfig) -> Self {
        // Window, computed in f64 like knf and rounded to f32.
        let a = match cfg.window {
            WindowKind::Hann => 2.0 * std::f64::consts::PI / WIN_LEN as f64,
            WindowKind::Povey => 2.0 * std::f64::consts::PI / (WIN_LEN as f64 - 1.0),
        };
        let window: Vec<f32> = (0..WIN_LEN)
            .map(|i| {
                let h = 0.5 - 0.5 * (a * i as f64).cos();
                (match cfg.window {
                    WindowKind::Hann => h,
                    WindowKind::Povey => h.powf(0.85),
                }) as f32
            })
            .collect();

        // knf InitLibrosaMelBanks (is_librosa=true, slaney mel scale, norm="slaney").
        let nyquist = SAMPLE_RATE as f32 / 2.0;
        let low_freq = 0.0f32;
        let high_freq = if cfg.high_freq > 0.0 { cfg.high_freq } else { nyquist + cfg.high_freq };
        let fft_bin_width = SAMPLE_RATE as f32 / FFT_LEN as f32;
        let mel_low = mel_scale_slaney(low_freq);
        let mel_high = mel_scale_slaney(high_freq);
        let delta = (mel_high - mel_low) / (FEAT_DIM + 1) as f32;
        let mut mel = Vec::with_capacity(FEAT_DIM);
        for bin in 0..FEAT_DIM {
            let left = inv_mel_scale_slaney(mel_low + bin as f32 * delta);
            let center = inv_mel_scale_slaney(mel_low + (bin + 1) as f32 * delta);
            let right = inv_mel_scale_slaney(mel_low + (bin + 2) as f32 * delta);
            let mut first: Option<usize> = None;
            let mut last = 0usize;
            let mut w = vec![0.0f32; N_BINS];
            for (i, wi) in w.iter_mut().enumerate() {
                let hz = fft_bin_width * i as f32;
                if hz > left && hz < right {
                    let mut weight = if hz <= center { (hz - left) / (center - left) } else { (right - hz) / (right - center) };
                    weight *= 2.0 / (right - left);
                    *wi = weight;
                    if first.is_none() {
                        first = Some(i);
                    }
                    last = i;
                }
            }
            let first = first.expect("mel bin without FFT support");
            mel.push((first, w[first..=last].to_vec()));
        }
        Self { cfg, window, mel, fft: Fft::new(FFT_LEN) }
    }

    pub fn config(&self) -> &FbankConfig {
        &self.cfg
    }

    /// Log-mel features, row-major `[frames, 128]`, BEFORE normalisation. The configured tail padding is appended
    /// internally. Returns an empty vector when the signal is too short for a single frame.
    pub fn compute(&self, audio: &[f32]) -> Vec<f32> {
        let mut padded;
        let wave: &[f32] = if self.cfg.tail_pad_samples > 0 {
            padded = Vec::with_capacity(audio.len() + self.cfg.tail_pad_samples);
            padded.extend_from_slice(audio);
            padded.resize(audio.len() + self.cfg.tail_pad_samples, 0.0);
            &padded
        } else {
            audio
        };
        let frames = self.cfg.num_frames(audio.len());
        let mut out = vec![0.0f32; frames * FEAT_DIM];
        if frames == 0 || wave.is_empty() {
            return Vec::new();
        }
        let n = wave.len() as i64;
        let mut re = vec![0.0f32; FFT_LEN];
        let mut im = vec![0.0f32; FFT_LEN];
        let mut power = vec![0.0f32; N_BINS];
        for f in 0..frames {
            let start: i64 = if self.cfg.snip_edges {
                (f * WIN_SHIFT) as i64
            } else {
                (WIN_SHIFT * f + WIN_SHIFT / 2) as i64 - (WIN_LEN / 2) as i64
            };
            for s in 0..WIN_LEN {
                let mut idx = start + s as i64;
                while idx < 0 || idx >= n {
                    idx = if idx < 0 { -idx - 1 } else { 2 * n - 1 - idx };
                }
                re[s] = wave[idx as usize];
            }
            // knf: pre-emphasis (0.97) on the 400 samples, then the window; the FFT frame is zero padded to 512.
            for i in (1..WIN_LEN).rev() {
                re[i] -= 0.97 * re[i - 1];
            }
            re[0] -= 0.97 * re[0];
            for i in 0..WIN_LEN {
                re[i] *= self.window[i];
            }
            for v in re[WIN_LEN..].iter_mut() {
                *v = 0.0;
            }
            for v in im.iter_mut() {
                *v = 0.0;
            }
            self.fft.forward(&mut re, &mut im);
            for k in 0..N_BINS {
                power[k] = re[k] * re[k] + im[k] * im[k];
            }
            let row = &mut out[f * FEAT_DIM..(f + 1) * FEAT_DIM];
            for (m, (first, w)) in self.mel.iter().enumerate() {
                let mut e = 0.0f32;
                for (j, wj) in w.iter().enumerate() {
                    e += wj * power[first + j];
                }
                row[m] = match self.cfg.log_guard {
                    LogGuard::Floor => e.max(f32::EPSILON).ln(),
                    LogGuard::Additive => (e + f32::EPSILON / 2.0).ln(),
                };
            }
        }
        out
    }
}

/// NeMo `per_feature` normalisation, in place on row-major `[frames, 128]`: `(x - mean) / (std + 1e-5)` with the
/// UNBIASED std over all frames (NeMo `normalize_batch` and `test_onnx.py`; sherpa-onnx's C++ uses the biased std,
/// the difference is a factor sqrt(T/(T-1)) < 0.5% for T >= 100).
pub fn normalize_per_feature(feats: &mut [f32], frames: usize) {
    if frames == 0 {
        return;
    }
    let denom = if frames > 1 { (frames - 1) as f64 } else { 1.0 };
    for c in 0..FEAT_DIM {
        let mut mean = 0.0f64;
        for t in 0..frames {
            mean += feats[t * FEAT_DIM + c] as f64;
        }
        mean /= frames as f64;
        let mut var = 0.0f64;
        for t in 0..frames {
            let d = feats[t * FEAT_DIM + c] as f64 - mean;
            var += d * d;
        }
        let std = (var / denom).sqrt() as f32 + NORM_EPS;
        for t in 0..frames {
            let v = &mut feats[t * FEAT_DIM + c];
            *v = (*v - mean as f32) / std;
        }
    }
}

/// `[frames, 128]` row-major → `[128, frames]` (the encoder input layout `[1, 128, T]`).
pub fn to_channel_major(feats: &[f32], frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; feats.len()];
    for t in 0..frames {
        for c in 0..FEAT_DIM {
            out[c * frames + t] = feats[t * FEAT_DIM + c];
        }
    }
    out
}

/// Summary statistics of raw (un-normalised) features, for the diagnostic log.
pub fn feature_summary(feats: &[f32], frames: usize) -> String {
    if frames == 0 {
        return "no frames".into();
    }
    let (mut mn, mut mx, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
    for &v in feats {
        mn = mn.min(v);
        mx = mx.max(v);
        sum += v as f64;
    }
    format!("raw log-mel min={:.2} max={:.2} mean={:.2}", mn, mx, sum / feats.len() as f64)
}

// ───────────────────────────────────────────────────────────── Vocabulary ────────────────────────

/// Parsed and validated `tokens.txt`.
#[derive(Clone, Debug)]
pub struct Vocab {
    pub tokens: Vec<String>,
    pub blank_id: usize,
}

impl Vocab {
    /// Strict parser. The sherpa-onnx test script does `t, idx = line.split()` and `blank = len(tokens) - 1`; this
    /// parser enforces the same contract and turns every deviation into an error:
    ///   * each non-empty line has exactly two whitespace separated fields `<token> <id>`;
    ///   * ids are integers, unique, and form the contiguous range `0..N`;
    ///   * the token with id `N-1` is `<blk>` and no other token is `<blk>` (so `blank_id == N-1`).
    pub fn parse_strict(text: &str) -> Result<Self, String> {
        let mut entries: Vec<(usize, String)> = Vec::new();
        for (ln, line) in text.lines().enumerate() {
            let line = line.trim_end_matches('\r');
            if line.trim().is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() != 2 {
                return Err(format!("tokens.txt line {}: expected '<token> <id>', got {} field(s): {:?}", ln + 1, fields.len(), line));
            }
            let id: usize = fields[1].parse().map_err(|_| format!("tokens.txt line {}: id {:?} is not a non-negative integer", ln + 1, fields[1]))?;
            entries.push((id, fields[0].to_string()));
        }
        if entries.is_empty() {
            return Err("tokens.txt is empty".into());
        }
        let n = entries.len();
        let mut tokens: Vec<Option<String>> = vec![None; n];
        for (id, tok) in entries {
            if id >= n {
                return Err(format!("tokens.txt: id {id} is outside 0..{n} (ids must be contiguous)"));
            }
            if tokens[id].is_some() {
                return Err(format!("tokens.txt: duplicate id {id}"));
            }
            tokens[id] = Some(tok);
        }
        let tokens: Vec<String> = tokens.into_iter().map(|t| t.expect("contiguity checked above")).collect();
        let blank_id = n - 1;
        if tokens[blank_id] != "<blk>" {
            return Err(format!("tokens.txt: the last token (id {blank_id}) must be <blk>, found {:?}", tokens[blank_id]));
        }
        if let Some(pos) = tokens.iter().position(|t| t == "<blk>").filter(|&p| p != blank_id) {
            return Err(format!("tokens.txt: <blk> appears at id {pos} as well as at id {blank_id}"));
        }
        Ok(Self { tokens, blank_id })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Control tokens (`<unk>`, `<pad>`, `<|ru|>`, `<blk>` …) never carry text.
    pub fn is_control(&self, id: usize) -> bool {
        let t = &self.tokens[id];
        (t.starts_with("<|") && t.ends_with("|>")) || t == "<unk>" || t == "<pad>" || t == "<blk>"
    }

    /// Sentence-piece detokenisation: concatenate, `▁` → space, trim. Returns the text and the number of control
    /// tokens that were skipped.
    pub fn detokenize(&self, ids: &[usize]) -> (String, usize) {
        let mut s = String::new();
        let mut skipped = 0;
        for &id in ids {
            if id >= self.tokens.len() {
                continue;
            }
            if self.is_control(id) {
                skipped += 1;
                continue;
            }
            s.push_str(&self.tokens[id]);
        }
        (s.replace('\u{2581}', " ").trim().to_string(), skipped)
    }
}

// ───────────────────────────────────────────────────────────── Transducer loop ──────────────────

/// What `transcribe` needs to know about the loaded model. Filled (and validated) by the ONNX layer.
#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub vocab: Vocab,
    pub pred_layers: usize,
    pub pred_hidden: usize,
    /// `normalize_type == "per_feature"` in the encoder metadata.
    pub per_feature_norm: bool,
}

pub struct Encoded {
    /// Channel major `[1024, t_enc]`.
    pub data: Vec<f32>,
    pub t_enc: usize,
}

pub struct DecoderOut {
    /// Prediction network output, `pred_hidden` floats.
    pub out: Vec<f32>,
    pub s0: Vec<f32>,
    pub s1: Vec<f32>,
}

/// Three model calls. In the app they are ONNX Runtime sessions; in the tests a reference process.
pub trait Backend {
    /// `feats_ct`: channel major `[128, frames]` (already normalised).
    fn encode(&mut self, feats_ct: &[f32], frames: usize) -> Result<Encoded, String>;
    /// token `i32` (blank at the start), states `[layers * 1 * hidden]` each.
    fn decoder(&mut self, token: i32, s0: &[f32], s1: &[f32]) -> Result<DecoderOut, String>;
    /// `enc_frame`: 1024 floats, `dec_out`: `pred_hidden` floats → `vocab + N_DURATIONS` logits.
    fn joiner(&mut self, enc_frame: &[f32], dec_out: &[f32]) -> Result<Vec<f32>, String>;
}

/// How a zero duration is handled in the greedy TDT loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TdtRule {
    /// sherpa-onnx `test_onnx.py`: `skip = argmax(durations); if skip == 0: skip = 1` for every token. DEFAULT — it is
    /// the reference that was reproduced on the real model.
    Sherpa,
    /// NeMo `GreedyTDTInfer` frame handling: a non-blank token with duration 0 stays on the same frame; after
    /// `max_symbols` tokens on one frame the frame advances by one. Kept for A/B only.
    Nemo { max_symbols: usize },
}

impl TdtRule {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "sherpa" | "" => Some(Self::Sherpa),
            "nemo" => Some(Self::Nemo { max_symbols: 10 }),
            _ => None,
        }
    }
    pub fn describe(&self) -> String {
        match self {
            Self::Sherpa => "tdt=sherpa (zero skip -> 1)".into(),
            Self::Nemo { max_symbols } => format!("tdt=nemo (zero skip stays on frame, max {max_symbols} symbols/frame)"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Variant {
    pub fbank: FbankConfig,
    pub tdt: TdtRule,
}

impl Variant {
    pub fn reference() -> Self {
        Self { fbank: FbankConfig::release(), tdt: TdtRule::Sherpa }
    }
}

/// Official Microsoft ONNX Runtime 1.24.1 / 1.24.2 builds, recognised by the commit id in `ort::info()`
/// (`GetBuildInfoString` carries no version number). Measured on the real int8 model: both return empty text for short
/// fragments that every ORT >= 1.25.0 transcribes. Unknown builds are NOT treated as legacy.
pub fn is_legacy_ort(build_info: &str) -> bool {
    build_info.contains("git-commit-id=058787ceea") || build_info.contains("git-commit-id=470ae16099")
}

/// Front end / TDT rule for a stream. `env_*` are the values of NEXQ_PARAKEET_VARIANT / NEXQ_PARAKEET_TDT.
/// Default: `release`; on a legacy ORT build `test_onnx` (measured: the only recipe that stays usable there);
/// an explicit variant always wins. Returns the log lines to emit (`level`, `message`).
pub fn select_variant(env_variant: Option<&str>, env_tdt: Option<&str>, legacy_ort: bool) -> (Variant, Vec<(&'static str, String)>) {
    let mut v = Variant::reference();
    let mut log = Vec::new();
    let mut explicit = false;
    if let Some(name) = env_variant {
        match FbankConfig::from_name(name) {
            Some(cfg) => {
                v.fbank = cfg;
                explicit = true;
                log.push(("info", format!("front end set by NEXQ_PARAKEET_VARIANT={name}")));
            }
            None => log.push(("warn", format!("NEXQ_PARAKEET_VARIANT={name:?} is not a known Fbank variant (release, test_onnx, master, hann_nopad); using the default"))),
        }
    }
    if !explicit && legacy_ort {
        v.fbank = FbankConfig::test_onnx();
        log.push(("info", "front end auto-selected: test_onnx (known official ONNX Runtime 1.24.x build); override with NEXQ_PARAKEET_VARIANT".to_string()));
    }
    if let Some(name) = env_tdt {
        match TdtRule::from_name(name) {
            Some(rule) => {
                v.tdt = rule;
                log.push(("info", format!("TDT rule set by NEXQ_PARAKEET_TDT={name}")));
            }
            None => log.push(("warn", format!("NEXQ_PARAKEET_TDT={name:?} is not a known TDT rule (sherpa, nemo); using the default (sherpa)"))),
        }
    }
    (v, log)
}

#[derive(Clone, Debug, Default)]
pub struct DecodeStats {
    pub audio_samples: usize,
    pub feature_frames: usize,
    pub encoder_frames: usize,
    pub joiner_calls: usize,
    pub tokens: usize,
    pub control_tokens_skipped: usize,
    pub zero_skips: usize,
    pub nonblank_zero_skips: usize,
    /// How often each duration index was predicted.
    pub duration_hist: [usize; N_DURATIONS],
    pub feature_summary: String,
}

#[derive(Clone, Debug)]
pub struct Decoded {
    pub token_ids: Vec<usize>,
    pub text: String,
    pub stats: DecodeStats,
}

fn argmax(xs: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in xs.iter().enumerate() {
        if x > xs[best] {
            best = i;
        }
    }
    best
}

/// Audio (f32, 16 kHz mono) → text, through `backend`.
pub fn transcribe<B: Backend>(backend: &mut B, info: &ModelInfo, fbank: &Fbank, tdt: TdtRule, audio: &[f32]) -> Result<Decoded, String> {
    let mut stats = DecodeStats { audio_samples: audio.len(), ..Default::default() };
    let mut feats = fbank.compute(audio);
    let frames = feats.len() / FEAT_DIM;
    stats.feature_frames = frames;
    if frames == 0 {
        return Ok(Decoded { token_ids: vec![], text: String::new(), stats });
    }
    stats.feature_summary = feature_summary(&feats, frames);
    if info.per_feature_norm {
        normalize_per_feature(&mut feats, frames);
    }
    let ct = to_channel_major(&feats, frames);
    let enc = backend.encode(&ct, frames)?;
    if enc.t_enc == 0 || enc.data.len() != ENC_DIM * enc.t_enc {
        return Err(format!("encoder returned {} floats for t_enc={} (expected {})", enc.data.len(), enc.t_enc, ENC_DIM * enc.t_enc));
    }
    stats.encoder_frames = enc.t_enc;

    let vocab_len = info.vocab.len();
    let blank = info.vocab.blank_id;
    let state_len = info.pred_layers * info.pred_hidden;
    let check_dec = |d: &DecoderOut| -> Result<(), String> {
        if d.out.len() != info.pred_hidden || d.s0.len() != state_len || d.s1.len() != state_len {
            return Err(format!(
                "decoder returned out={} s0={} s1={}, expected {} / {} / {}",
                d.out.len(), d.s0.len(), d.s1.len(), info.pred_hidden, state_len, state_len
            ));
        }
        Ok(())
    };

    // Decoder primed with the blank token and zero states (sherpa-onnx `test_onnx.py`).
    let mut s0 = vec![0.0f32; state_len];
    let mut s1 = vec![0.0f32; state_len];
    let mut dec = backend.decoder(blank as i32, &s0, &s1)?;
    check_dec(&dec)?;

    let mut ids: Vec<usize> = Vec::new();
    let mut frame = vec![0.0f32; ENC_DIM];
    let mut t = 0usize;
    let mut emitted_here = 0usize;
    while t < enc.t_enc {
        for c in 0..ENC_DIM {
            frame[c] = enc.data[c * enc.t_enc + t];
        }
        let logits = backend.joiner(&frame, &dec.out)?;
        stats.joiner_calls += 1;
        if logits.len() != vocab_len + N_DURATIONS {
            return Err(format!("joiner returned {} logits, expected vocab {} + {} durations", logits.len(), vocab_len, N_DURATIONS));
        }
        let token = argmax(&logits[..vocab_len]);
        let mut skip = argmax(&logits[vocab_len..]);
        stats.duration_hist[skip] += 1;
        if skip == 0 {
            stats.zero_skips += 1;
            if token != blank {
                stats.nonblank_zero_skips += 1;
            }
        }
        match tdt {
            TdtRule::Sherpa => {
                if skip == 0 {
                    skip = 1;
                }
                if token != blank {
                    ids.push(token);
                    // Commit the states produced by the *previous* decoder call, then run the decoder on the new token.
                    s0 = std::mem::take(&mut dec.s0);
                    s1 = std::mem::take(&mut dec.s1);
                    dec = backend.decoder(token as i32, &s0, &s1)?;
                    check_dec(&dec)?;
                }
                t += skip;
            }
            TdtRule::Nemo { max_symbols } => {
                if token == blank {
                    skip = skip.max(1);
                } else {
                    ids.push(token);
                    s0 = std::mem::take(&mut dec.s0);
                    s1 = std::mem::take(&mut dec.s1);
                    dec = backend.decoder(token as i32, &s0, &s1)?;
                    check_dec(&dec)?;
                    emitted_here += 1;
                    if skip == 0 && emitted_here >= max_symbols {
                        skip = 1;
                    }
                }
                if skip > 0 {
                    emitted_here = 0;
                    t += skip;
                }
            }
        }
    }
    stats.tokens = ids.len();
    let (text, skipped) = info.vocab.detokenize(&ids);
    stats.control_tokens_skipped = skipped;
    Ok(Decoded { token_ids: ids, text, stats })
}

// ───────────────────────────────────────────────────────────── Segmenter ────────────────────────

/// Segmenter parameters. The numbers are those of revision 1 of the patch (kept, not tuned); the tests in this file
/// show where they work and where they do not.
#[derive(Clone, Debug)]
pub struct SegmenterConfig {
    pub start_rms: f32,
    pub continue_rms: f32,
    pub silence_samples: usize,
    pub max_samples: usize,
    pub min_samples: usize,
    pub preroll_samples: usize,
    /// When `max_samples` is reached the cut is moved to the quietest 100 ms of the last `cut_search_samples`
    /// instead of an arbitrary point (which falls inside a word and, measured on the real model, leaves the
    /// remainder as an abrupt fragment the int8 encoder often cannot transcribe).
    pub cut_search_samples: usize,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            start_rms: 0.004,
            continue_rms: 0.0025,
            silence_samples: SAMPLE_RATE * 8 / 10,
            max_samples: SAMPLE_RATE * 10,
            min_samples: SAMPLE_RATE * 3 / 10,
            preroll_samples: SAMPLE_RATE / 5,
            cut_search_samples: SAMPLE_RATE * 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndReason {
    /// `silence_samples` of quiet after speech.
    Pause,
    /// `max_samples` reached; cut at the quietest point of the last `cut_search_samples`.
    MaxLength,
    /// The stream stopped; the remaining audio is flushed.
    Stop,
}

#[derive(Clone, Debug)]
pub enum SegEvent {
    /// Speech began; `at_sample` is the stream position of the first sample of the segment (preroll included).
    Started { at_sample: u64, rms: f32 },
    /// A finished segment ready for the model.
    Segment { samples: Vec<f32>, start_sample: u64, reason: EndReason },
    /// A finished segment that was NOT sent to the model, with the reason (never silent).
    Dropped { len: usize, reason: &'static str },
}

pub struct Segmenter {
    cfg: SegmenterConfig,
    pending: Vec<f32>,
    preroll: VecDeque<f32>,
    segment: Vec<f32>,
    in_segment: bool,
    silence_run: usize,
    consumed: u64,
    segment_start: u64,
    pub blocks_seen: u64,
    pub blocks_above_start: u64,
    pub max_block_rms: f32,
}

impl Segmenter {
    pub fn new(cfg: SegmenterConfig) -> Self {
        Self {
            cfg,
            pending: Vec::new(),
            preroll: VecDeque::new(),
            segment: Vec::new(),
            in_segment: false,
            silence_run: 0,
            consumed: 0,
            segment_start: 0,
            blocks_seen: 0,
            blocks_above_start: 0,
            max_block_rms: 0.0,
        }
    }

    pub fn config(&self) -> &SegmenterConfig {
        &self.cfg
    }
    pub fn in_segment(&self) -> bool {
        self.in_segment
    }
    pub fn segment_len(&self) -> usize {
        self.segment.len()
    }

    fn rms(block: &[f32]) -> f32 {
        (block.iter().map(|x| x * x).sum::<f32>() / block.len().max(1) as f32).sqrt()
    }

    fn finish_segment(&mut self, reason: EndReason, events: &mut Vec<SegEvent>) {
        let samples = std::mem::take(&mut self.segment);
        // The minimum applies to the audio that is NOT the trailing silence that ended the segment (preroll counts):
        // with `silence_samples` (0.8 s) > `min_samples` (0.3 s) a check on the whole segment could never fire.
        let speech_len = samples.len().saturating_sub(self.silence_run);
        self.in_segment = false;
        self.silence_run = 0;
        self.preroll.clear();
        let enough = match reason {
            // At stop only the model's own limit matters: at least one 10 ms frame.
            EndReason::Stop => samples.len() >= WIN_SHIFT,
            _ => speech_len >= self.cfg.min_samples,
        };
        if enough {
            events.push(SegEvent::Segment { samples, start_sample: self.segment_start, reason });
        } else {
            events.push(SegEvent::Dropped { len: speech_len, reason: "speech part shorter than the minimum segment length" });
        }
    }

    /// Length limit reached while speech continues: end the segment at the quietest 100 ms window of the last
    /// `cut_search_samples` (latest one on ties) and keep the rest as the beginning of the next segment, so that no
    /// sample is lost or duplicated and the cut does not fall in the middle of a word when there is any dip.
    fn cut_at_quiet_point(&mut self, events: &mut Vec<SegEvent>) {
        const WINDOW: usize = SAMPLE_RATE / 10;
        let len = self.segment.len();
        // both parts must stay at least `min_samples` long
        let reach = self.cfg.cut_search_samples.min(len.saturating_sub(self.cfg.min_samples));
        if reach < WINDOW {
            self.finish_segment(EndReason::MaxLength, events);
            return;
        }
        let first = len - reach;
        let last = len - WINDOW;
        let mut best = (f32::MAX, last);
        let mut i = first;
        while i <= last {
            let e: f32 = self.segment[i..i + WINDOW].iter().map(|x| x * x).sum();
            if e <= best.0 {
                best = (e, i);
            }
            i += WIN_SHIFT;
        }
        let cut = best.1 + WINDOW / 2;
        let rest = self.segment.split_off(cut);
        let head = std::mem::replace(&mut self.segment, rest);
        events.push(SegEvent::Segment { samples: head, start_sample: self.segment_start, reason: EndReason::MaxLength });
        self.segment_start += cut as u64;
        // the quiet tail of what is carried over counts for the pause rule
        self.silence_run = 0;
        for block in self.segment.rchunks_exact(WIN_SHIFT) {
            if Self::rms(block) > self.cfg.continue_rms {
                break;
            }
            self.silence_run += WIN_SHIFT;
        }
    }

    /// Feed mono 16 kHz samples (`[-1, 1]`). Works on 10 ms blocks, independent of the sender's chunk size.
    pub fn push(&mut self, samples: &[f32]) -> Vec<SegEvent> {
        let mut events = Vec::new();
        self.pending.extend_from_slice(samples);
        let complete = self.pending.len() / WIN_SHIFT * WIN_SHIFT;
        let blocks: Vec<f32> = self.pending.drain(..complete).collect();
        for block in blocks.chunks_exact(WIN_SHIFT) {
            let rms = Self::rms(block);
            self.blocks_seen += 1;
            self.max_block_rms = self.max_block_rms.max(rms);
            if rms > self.cfg.start_rms {
                self.blocks_above_start += 1;
            }
            let block_start = self.consumed;
            self.consumed += WIN_SHIFT as u64;
            if !self.in_segment {
                if rms > self.cfg.start_rms {
                    self.segment.extend(self.preroll.drain(..));
                    self.segment_start = block_start - self.segment.len() as u64;
                    self.segment.extend_from_slice(block);
                    self.in_segment = true;
                    self.silence_run = 0;
                    events.push(SegEvent::Started { at_sample: self.segment_start, rms });
                } else {
                    self.preroll.extend(block.iter().copied());
                    while self.preroll.len() > self.cfg.preroll_samples {
                        self.preroll.pop_front();
                    }
                }
            } else {
                self.segment.extend_from_slice(block);
                if rms > self.cfg.continue_rms {
                    self.silence_run = 0;
                } else {
                    self.silence_run += block.len();
                }
                if self.silence_run >= self.cfg.silence_samples {
                    self.finish_segment(EndReason::Pause, &mut events);
                } else if self.segment.len() >= self.cfg.max_samples {
                    self.cut_at_quiet_point(&mut events);
                }
            }
        }
        events
    }

    /// End of stream: flush the open segment including the < 10 ms remainder.
    pub fn finish(&mut self) -> Vec<SegEvent> {
        let mut events = Vec::new();
        if self.in_segment {
            let rest = std::mem::take(&mut self.pending);
            self.segment.extend_from_slice(&rest);
            self.finish_segment(EndReason::Stop, &mut events);
        } else {
            self.pending.clear();
        }
        events
    }
}

// ───────────────────────────────────────────────────────────── AudioMeter ───────────────────────

/// Counters for the diagnostic log: how much audio really arrived and how loud it is.
#[derive(Clone, Debug, Default)]
pub struct AudioMeter {
    pub chunks: u64,
    pub samples: u64,
    pub clipped: u64,
    pub peak: f32,
    sum_sq: f64,
    win_samples: u64,
    win_sum_sq: f64,
    win_peak: f32,
}

impl AudioMeter {
    pub fn add_i16(&mut self, chunk: &[i16]) {
        self.chunks += 1;
        for &s in chunk {
            let x = s as f32 / 32768.0;
            let a = x.abs();
            self.samples += 1;
            self.sum_sq += (x as f64) * (x as f64);
            self.win_samples += 1;
            self.win_sum_sq += (x as f64) * (x as f64);
            if a > self.peak {
                self.peak = a;
            }
            if a > self.win_peak {
                self.win_peak = a;
            }
            if s == i16::MIN || s == i16::MAX {
                self.clipped += 1;
            }
        }
    }
    pub fn total_rms(&self) -> f32 {
        if self.samples == 0 { 0.0 } else { (self.sum_sq / self.samples as f64).sqrt() as f32 }
    }
    /// RMS and peak since the last call; resets the window.
    pub fn take_window(&mut self) -> (u64, f32, f32) {
        let n = self.win_samples;
        let rms = if n == 0 { 0.0 } else { (self.win_sum_sq / n as f64).sqrt() as f32 };
        let peak = self.win_peak;
        self.win_samples = 0;
        self.win_sum_sq = 0.0;
        self.win_peak = 0.0;
        (n, rms, peak)
    }
}

pub fn dbfs(x: f32) -> f32 {
    20.0 * x.max(1e-9).log10()
}

// ───────────────────────────────────────────────────────────── Driver ───────────────────────────
//
// The receive → segment → decode → emit loop, independent of the application's channel types. The glue passes
// closures for "next audio message", "stop requested?" and "deliver a result"; the tests pass a real model behind
// `Backend` and a scripted audio stream. Everything the loop does is reported through `debug`.

pub fn epoch_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

pub fn panic_text(p: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".into()
    }
}

/// 16 kHz mono PCM16 WAV (used for the optional per-segment dump).
pub fn write_wav_pcm16(path: &std::path::Path, samples: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let data_len = (samples.len() * 2) as u32;
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&1u16.to_le_bytes())?; // mono
    f.write_all(&(SAMPLE_RATE as u32).to_le_bytes())?;
    f.write_all(&((SAMPLE_RATE * 2) as u32).to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    for &s in samples {
        f.write_all(&((s * 32768.0).round().clamp(-32768.0, 32767.0) as i16).to_le_bytes())?;
    }
    f.flush()
}

/// Reads a 16 kHz PCM16 WAV file (first channel when stereo) into f32 samples in [-1, 1).
/// Used by the real-model tests; a wrong format is an error, never a silent resample.
pub fn read_wav_mono16k(path: &std::path::Path) -> Result<Vec<f32>, String> {
    let d = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if d.len() < 12 || &d[0..4] != b"RIFF" || &d[8..12] != b"WAVE" {
        return Err(format!("not a WAV file: {}", path.display()));
    }
    let (mut pos, mut ch, mut bits, mut rate, mut fmt_tag) = (12usize, 0u16, 0u16, 0u32, 0u16);
    while pos + 8 <= d.len() {
        let id = &d[pos..pos + 4];
        let sz = u32::from_le_bytes([d[pos + 4], d[pos + 5], d[pos + 6], d[pos + 7]]) as usize;
        if id == b"fmt " && pos + 8 + 16 <= d.len() {
            fmt_tag = u16::from_le_bytes([d[pos + 8], d[pos + 9]]);
            ch = u16::from_le_bytes([d[pos + 10], d[pos + 11]]);
            rate = u32::from_le_bytes([d[pos + 12], d[pos + 13], d[pos + 14], d[pos + 15]]);
            bits = u16::from_le_bytes([d[pos + 22], d[pos + 23]]);
        } else if id == b"data" {
            if fmt_tag != 1 || bits != 16 || rate != SAMPLE_RATE as u32 || ch < 1 {
                return Err(format!(
                    "need PCM16 16 kHz WAV, got format={fmt_tag} bits={bits} rate={rate} channels={ch}: {}",
                    path.display()
                ));
            }
            let end = (pos + 8 + sz).min(d.len());
            return Ok(d[pos + 8..end]
                .chunks_exact(2 * ch as usize)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                .collect());
        }
        pos += 8 + sz + (sz & 1);
    }
    Err(format!("no data chunk in {}", path.display()))
}

fn levenshtein<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, y) in b.iter().enumerate() {
            cur.push((prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + usize::from(x != y)));
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Lower-case, `ё`→`е`, punctuation removed, whitespace collapsed (for comparing recognised text).
pub fn normalize_text_for_compare(s: &str) -> String {
    let l: String = s
        .to_lowercase()
        .replace('ё', "е")
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect();
    l.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Character error rate of `hyp` against `reference` after `normalize_text_for_compare`.
pub fn text_cer(hyp: &str, reference: &str) -> f32 {
    let (h, r) = (normalize_text_for_compare(hyp), normalize_text_for_compare(reference));
    let (hc, rc): (Vec<char>, Vec<char>) = (h.chars().collect(), r.chars().collect());
    levenshtein(&hc, &rc) as f32 / rc.len().max(1) as f32
}

/// One poll of the audio source.
pub enum RecvEvent {
    Samples(Vec<i16>),
    /// Explicit stop message from the application.
    Stop,
    /// Nothing arrived within the timeout.
    Timeout,
    /// The sender is gone.
    Disconnected,
}

/// A finished, recognised phrase.
#[derive(Clone, Debug)]
pub struct Emission {
    pub text: String,
    pub segment_id: String,
    /// Wall-clock start of the phrase (epoch ms): the segment was complete `duration_ms` after it.
    pub timestamp_ms: u64,
    pub duration_ms: u64,
}

pub struct DriverOptions {
    pub variant: Variant,
    pub segmenter: SegmenterConfig,
    /// The first `segment_id` is `ort_<segment_counter_start>`.
    pub segment_counter_start: u64,
    /// When set, every segment sent to the model is also written here as a WAV.
    pub dump_dir: Option<std::path::PathBuf>,
    /// How often the audio statistics line is logged.
    pub stats_interval: std::time::Duration,
}

impl DriverOptions {
    pub fn new(variant: Variant, segment_counter_start: u64) -> Self {
        Self {
            variant,
            segmenter: SegmenterConfig::default(),
            segment_counter_start,
            dump_dir: None,
            stats_interval: std::time::Duration::from_secs(10),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DriverReport {
    pub chunks: u64,
    pub samples: u64,
    pub clipped: u64,
    pub segments_started: u64,
    pub segments_decoded: u64,
    pub segments_empty: u64,
    pub segments_failed: u64,
    pub segments_dropped: u64,
    pub results_sent: u64,
    pub send_errors: u64,
}

/// Runs a segment through the model and delivers the text. Returns `false` when the receiver is gone.
#[allow(clippy::too_many_arguments)]
fn handle_events<B: Backend>(
    events: Vec<SegEvent>,
    backend: &mut B,
    info: &ModelInfo,
    fbank: &Fbank,
    tdt: TdtRule,
    dump_dir: Option<&std::path::Path>,
    seg_id: &mut u64,
    rep: &mut DriverReport,
    emit: &mut dyn FnMut(Emission) -> Result<(), String>,
    debug: &dyn Fn(&str, &str),
) -> bool {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let secs = |n: usize| n as f32 / SAMPLE_RATE as f32;
    for ev in events {
        match ev {
            SegEvent::Started { at_sample, rms } => {
                rep.segments_started += 1;
                debug("info", &format!("segment started at {:.2}s of stream (block rms {:.4} = {:.1} dBFS)", at_sample as f32 / SAMPLE_RATE as f32, rms, dbfs(rms)));
            }
            SegEvent::Dropped { len, reason } => {
                rep.segments_dropped += 1;
                debug("info", &format!("segment dropped ({:.2}s): {reason}", secs(len)));
            }
            SegEvent::Segment { samples, start_sample, reason } => {
                let ended_ms = epoch_ms();
                let dur_ms = (samples.len() * 1000 / SAMPLE_RATE) as u64;
                let id = format!("ort_{}", *seg_id);
                *seg_id += 1;
                debug(
                    "info",
                    &format!("segment {id} ended ({reason:?}): {:.2}s starting at {:.2}s, decoding", secs(samples.len()), start_sample as f32 / SAMPLE_RATE as f32),
                );
                if let Some(dir) = dump_dir {
                    let path = dir.join(format!("{id}.wav"));
                    if let Err(e) = write_wav_pcm16(&path, &samples) {
                        debug("warn", &format!("cannot write {}: {e}", path.display()));
                    }
                }
                let t0 = std::time::Instant::now();
                let decoded = catch_unwind(AssertUnwindSafe(|| transcribe(backend, info, fbank, tdt, &samples)));
                let ms = t0.elapsed().as_millis();
                match decoded {
                    Err(p) => {
                        rep.segments_failed += 1;
                        debug("error", &format!("segment {id}: inference panicked: {}", panic_text(&p)));
                    }
                    Ok(Err(e)) => {
                        rep.segments_failed += 1;
                        debug("error", &format!("segment {id}: inference failed: {e}"));
                    }
                    Ok(Ok(d)) => {
                        rep.segments_decoded += 1;
                        let st = &d.stats;
                        let summary = format!(
                            "{} feature frames -> {} encoder frames, {} joiner calls, {} tokens ({} control skipped), durations {:?}, {}, {} ms",
                            st.feature_frames, st.encoder_frames, st.joiner_calls, st.tokens, st.control_tokens_skipped, st.duration_hist, st.feature_summary, ms
                        );
                        if d.text.is_empty() {
                            rep.segments_empty += 1;
                            debug("info", &format!("segment {id}: model returned NO text: {summary}"));
                            continue;
                        }
                        debug("info", &format!("segment {id}: {summary}; text: {}", d.text));
                        let emission = Emission { text: d.text, segment_id: id.clone(), timestamp_ms: ended_ms.saturating_sub(dur_ms), duration_ms: dur_ms };
                        match emit(emission) {
                            Ok(()) => {
                                rep.results_sent += 1;
                                debug("info", &format!("segment {id}: TranscriptResult sent"));
                            }
                            Err(e) => {
                                rep.send_errors += 1;
                                debug("error", &format!("segment {id}: TranscriptResult NOT delivered: {e}"));
                                return false;
                            }
                        }
                    }
                }
            }
        }
    }
    true
}

/// Runs until stop / disconnect / a failed delivery. `Err` only when the model fails its start-up self-test.
pub fn drive<B: Backend>(
    backend: &mut B,
    info: &ModelInfo,
    opts: DriverOptions,
    recv: &mut dyn FnMut(std::time::Duration) -> RecvEvent,
    stopped: &dyn Fn() -> bool,
    emit: &mut dyn FnMut(Emission) -> Result<(), String>,
    debug: &dyn Fn(&str, &str),
) -> Result<DriverReport, String> {
    use std::time::{Duration, Instant};
    let fbank = Fbank::new(opts.variant.fbank.clone());
    let tdt = opts.variant.tdt;
    let seg_cfg = opts.segmenter.clone();
    debug("info", &format!("frontend: {}", fbank.config().describe()));
    debug("info", &format!("decoder loop: {}", tdt.describe()));
    debug(
        "info",
        &format!(
            "segmenter (RMS, not a VAD): start>{:.4} continue>{:.4} pause={}ms max={}ms (cut at the quietest point of the last {}ms) min={}ms preroll={}ms",
            seg_cfg.start_rms,
            seg_cfg.continue_rms,
            seg_cfg.silence_samples * 1000 / SAMPLE_RATE,
            seg_cfg.max_samples * 1000 / SAMPLE_RATE,
            seg_cfg.cut_search_samples * 1000 / SAMPLE_RATE,
            seg_cfg.min_samples * 1000 / SAMPLE_RATE,
            seg_cfg.preroll_samples * 1000 / SAMPLE_RATE
        ),
    );

    // Self-test: one second of faint noise through all three sessions, so that a shape/type/state problem is reported
    // now and not after the first phrase. The text is not checked (it is expected to be empty).
    {
        let t0 = Instant::now();
        let warm: Vec<f32> = (0..SAMPLE_RATE).map(|i| ((i * 7919 % 2003) as f32 / 2003.0 - 0.5) * 0.004).collect();
        let d = transcribe(backend, info, &fbank, tdt, &warm).map_err(|e| format!("self-test failed: {e}"))?;
        debug(
            "info",
            &format!(
                "self-test OK in {} ms: {} feature frames -> {} encoder frames, {} joiner calls, {} tokens (faint noise, normally 0)",
                t0.elapsed().as_millis(), d.stats.feature_frames, d.stats.encoder_frames, d.stats.joiner_calls, d.stats.tokens
            ),
        );
    }
    if let Some(dir) = &opts.dump_dir {
        match std::fs::create_dir_all(dir) {
            Ok(()) => debug("info", &format!("segment WAV dump enabled: {}", dir.display())),
            Err(e) => debug("warn", &format!("cannot create the dump directory {}: {e}", dir.display())),
        }
    }
    debug("info", "NeMo Parakeet ready; waiting for audio");

    let mut segmenter = Segmenter::new(seg_cfg);
    let mut meter = AudioMeter::default();
    let mut rep = DriverReport::default();
    let mut seg_id = opts.segment_counter_start;
    let mut last_stats = Instant::now();
    let started = Instant::now();
    let mut first_audio_logged = false;
    let mut silence_warned = false;
    let mut alive = true;
    // `stop_stream` sets the stop flag and only then queues `Stop`, behind every chunk that was already sent. When the
    // flag is seen, the audio that is already queued is still processed (up to the explicit Stop, an empty queue or
    // the drain limit), so the last words of the meeting are not thrown away together with the thread.
    let mut draining_since: Option<Instant> = None;
    const DRAIN_LIMIT: Duration = Duration::from_secs(60);

    while alive {
        match recv(Duration::from_millis(100)) {
            RecvEvent::Samples(samples) => {
                if !first_audio_logged {
                    first_audio_logged = true;
                    debug("info", &format!("first audio chunk reached the NeMo thread: {} samples (expected 16 kHz mono i16)", samples.len()));
                }
                meter.add_i16(&samples);
                let f: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
                let events = segmenter.push(&f);
                alive = handle_events(events, backend, info, &fbank, tdt, opts.dump_dir.as_deref(), &mut seg_id, &mut rep, emit, debug);
            }
            RecvEvent::Stop | RecvEvent::Disconnected => break,
            RecvEvent::Timeout => {
                if draining_since.is_some() {
                    break; // queue is empty and a stop was requested
                }
            }
        }
        if draining_since.is_none() && stopped() {
            draining_since = Some(Instant::now());
            debug("info", "stop requested: processing the audio that is already queued, then flushing the open segment");
        }
        if draining_since.map_or(false, |t| t.elapsed() > DRAIN_LIMIT) {
            debug("warn", "stop requested but audio keeps arriving: drain limit reached, flushing now");
            break;
        }
        if last_stats.elapsed() >= opts.stats_interval {
            let secs = last_stats.elapsed().as_secs_f32();
            let (n, rms, peak) = meter.take_window();
            let rate = n as f32 / secs;
            debug(
                "info",
                &format!(
                    "audio last {:.0}s: {} samples = {:.0} Hz (16000 expected), rms {:.1} dBFS, peak {:.1} dBFS | total {} chunks, {} samples, clipped {} | \
                     segmenter: {}/{} blocks above start, max block rms {:.4}, in_segment={} ({:.1}s) | \
                     segments started={} decoded={} empty={} failed={} dropped={} | results sent={} send_errors={}",
                    secs, n, rate, dbfs(rms), dbfs(peak), meter.chunks, meter.samples, meter.clipped,
                    segmenter.blocks_above_start, segmenter.blocks_seen, segmenter.max_block_rms, segmenter.in_segment(),
                    segmenter.segment_len() as f32 / SAMPLE_RATE as f32,
                    rep.segments_started, rep.segments_decoded, rep.segments_empty, rep.segments_failed, rep.segments_dropped,
                    rep.results_sent, rep.send_errors
                ),
            );
            if !silence_warned && meter.peak == 0.0 && started.elapsed() > Duration::from_secs(10) {
                silence_warned = true;
                debug("warn", "audio reaching the NeMo thread is digital silence (peak = 0): check the capture device / mute state");
            }
            if n > 0 && opts.stats_interval >= Duration::from_secs(5) && !(14_000.0..=18_000.0).contains(&rate) {
                debug("warn", &format!("audio arrives at {rate:.0} samples/s, not 16000: the capture/resampler stage delivers the wrong rate or drops chunks"));
            }
            last_stats = Instant::now();
        }
    }

    // Stop / disconnect: flush the open segment so a cut-off last phrase is not lost.
    if alive {
        let events = segmenter.finish();
        handle_events(events, backend, info, &fbank, tdt, opts.dump_dir.as_deref(), &mut seg_id, &mut rep, emit, debug);
    }
    rep.chunks = meter.chunks;
    rep.samples = meter.samples;
    rep.clipped = meter.clipped;
    debug(
        "info",
        &format!(
            "NeMo stream ended: {} chunks / {} samples received, segments started={} decoded={} empty={} failed={} dropped={}, results sent={} send_errors={}",
            rep.chunks, rep.samples, rep.segments_started, rep.segments_decoded, rep.segments_empty, rep.segments_failed,
            rep.segments_dropped, rep.results_sent, rep.send_errors
        ),
    );
    Ok(rep)
}


// ───────────────────────────────────────────────────────────── Tests ────────────────────────────
//
// Two kinds of tests live here and the names say which is which:
//   * plain `#[test]`   — synthetic unit tests of the logic (shape/rule/state-machine checks). They do NOT prove that
//                         the model recognises speech.
//   * `#[ignore]` `real_*` — tests that use real WAV files, the real ONNX model (through a reference process built
//                         from `tools/ab/ort_ab.cpp`) and real `kaldi-native-fbank` dumps. See README for the commands.
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::path::{Path, PathBuf};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

    // ── helpers ──

    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 // uniform [-1, 1)
        }
    }

    fn tone(len: usize, amp: f32) -> Vec<f32> {
        (0..len).map(|i| amp * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin()).collect()
    }

    fn noise(len: usize, rms: f32, seed: u64) -> Vec<f32> {
        let mut g = Lcg(seed);
        // uniform [-1,1) has rms 1/sqrt(3)
        (0..len).map(|_| g.next_f32() * rms * 3.0f32.sqrt()).collect()
    }

    fn read_wav(path: &Path) -> Vec<f32> {
        read_wav_mono16k(path).unwrap_or_else(|e| panic!("{e}"))
    }

    fn cer(h: &str, r: &str) -> f32 {
        text_cer(h, r)
    }

    #[test]
    fn wav_roundtrip_and_text_cer_helpers() {
        let dir = std::env::temp_dir().join(format!("nexq_pk_wav_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.wav");
        let x = tone(1600, 0.5);
        write_wav_pcm16(&path, &x).unwrap();
        let y = read_wav_mono16k(&path).unwrap();
        assert_eq!(y.len(), x.len());
        assert!(x.iter().zip(&y).all(|(a, b)| (a - b).abs() < 1.0 / 16384.0));
        // a file that is not 16 kHz must be an error, not a silent resample
        let mut raw = std::fs::read(&path).unwrap();
        raw[24..28].copy_from_slice(&8000u32.to_le_bytes());
        std::fs::write(&path, &raw).unwrap();
        assert!(read_wav_mono16k(&path).unwrap_err().contains("16 kHz"));
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(text_cer("Привет, Мир!", "привет мир"), 0.0);
        assert_eq!(text_cer("Её", "ее"), 0.0);
        assert!((text_cer("abcd", "abce") - 0.25).abs() < 1e-6);
        assert_eq!(text_cer("", "abc"), 1.0);
    }

    // ── scripted backend for the loop tests (SYNTHETIC) ──

    struct Scripted {
        script: Vec<(usize, usize)>, // (token, duration index) per joiner call
        t_enc: usize,
        joiner_calls: usize,
        decoder_calls: Vec<(i32, f32)>, // (token, first element of s0 handed in)
        counter: f32,
    }
    impl Scripted {
        fn new(script: Vec<(usize, usize)>, t_enc: usize) -> Self {
            Self { script, t_enc, joiner_calls: 0, decoder_calls: vec![], counter: 0.0 }
        }
    }
    impl Backend for Scripted {
        fn encode(&mut self, _f: &[f32], _n: usize) -> Result<Encoded, String> {
            Ok(Encoded { data: vec![0.0; ENC_DIM * self.t_enc], t_enc: self.t_enc })
        }
        fn decoder(&mut self, token: i32, s0: &[f32], _s1: &[f32]) -> Result<DecoderOut, String> {
            self.decoder_calls.push((token, s0[0]));
            self.counter += 1.0;
            Ok(DecoderOut { out: vec![token as f32, 0.0], s0: vec![self.counter; 2], s1: vec![self.counter; 2] })
        }
        fn joiner(&mut self, _e: &[f32], _d: &[f32]) -> Result<Vec<f32>, String> {
            let (tok, dur) = self.script.get(self.joiner_calls).copied().unwrap_or((9, 4));
            self.joiner_calls += 1;
            let mut l = vec![0.0f32; 10 + N_DURATIONS];
            l[tok] = 1.0;
            l[10 + dur] = 1.0;
            Ok(l)
        }
    }

    fn tiny_info() -> ModelInfo {
        let text: String = (0..9).map(|i| format!("t{i} {i}\n")).chain(std::iter::once("<blk> 9\n".to_string())).collect();
        ModelInfo { vocab: Vocab::parse_strict(&text).unwrap(), pred_layers: 1, pred_hidden: 2, per_feature_norm: true }
    }

    // ── Fbank (SYNTHETIC checks of the building blocks) ──

    #[test]
    fn fft_matches_naive_dft() {
        let fft = Fft::new(16);
        let x: Vec<f32> = noise(16, 0.3, 7);
        let (mut re, mut im) = (x.clone(), vec![0.0f32; 16]);
        fft.forward(&mut re, &mut im);
        for k in 0..16 {
            let (mut sr, mut si) = (0.0f64, 0.0f64);
            for (n, &v) in x.iter().enumerate() {
                let a = -2.0 * std::f64::consts::PI * (k * n) as f64 / 16.0;
                sr += v as f64 * a.cos();
                si += v as f64 * a.sin();
            }
            assert!((re[k] as f64 - sr).abs() < 1e-5 && (im[k] as f64 - si).abs() < 1e-5, "bin {k}");
        }
    }

    #[test]
    fn frame_counts_follow_knf_formulas() {
        // Values measured with the real knf on the 180640-sample test WAV (tools/ab/knf_dump).
        assert_eq!(FbankConfig::test_onnx().num_frames(180640), 1327); // snip_edges=true, +2 s
        assert_eq!(FbankConfig::release().num_frames(180640), 1129); // snip_edges=false
        assert_eq!(FbankConfig::hann_nopad().num_frames(180640), 1127);
        assert_eq!(FbankConfig::release().num_frames(0), 0);
        assert_eq!(FbankConfig::hann_nopad().num_frames(399), 0);
    }

    #[test]
    fn hann_window_is_periodic_and_povey_is_symmetric() {
        let h = Fbank::new(FbankConfig::release());
        assert!(h.window[0].abs() < 1e-7 && (h.window[200] - 1.0).abs() < 1e-6);
        assert!((h.window[1] - h.window[399]).abs() < 1e-6, "periodic hann: w[i] == w[N-i]");
        let p = Fbank::new(FbankConfig::master());
        assert!((p.window[1] - p.window[398]).abs() < 1e-6, "povey is symmetric over N-1");
    }

    #[test]
    fn mel_bank_has_128_nonempty_slaney_normalised_rows() {
        let f = Fbank::new(FbankConfig::release());
        assert_eq!(f.mel.len(), 128);
        for (first, w) in &f.mel {
            assert!(!w.is_empty() && w.iter().any(|&x| x > 0.0) && first + w.len() <= N_BINS);
        }
        // low bins are narrow (area normalisation makes their peak weights large)
        assert!(f.mel[0].1.iter().cloned().fold(0.0, f32::max) > f.mel[127].1.iter().cloned().fold(0.0, f32::max));
    }

    #[test]
    fn features_are_finite_and_have_expected_shape() {
        for cfg in [FbankConfig::release(), FbankConfig::test_onnx(), FbankConfig::master(), FbankConfig::hann_nopad()] {
            let f = Fbank::new(cfg.clone());
            let x = noise(16000, 0.05, 3);
            let feats = f.compute(&x);
            assert_eq!(feats.len(), cfg.num_frames(16000) * FEAT_DIM, "{}", cfg.name);
            assert!(feats.iter().all(|v| v.is_finite()), "{}", cfg.name);
        }
        assert!(Fbank::new(FbankConfig::hann_nopad()).compute(&[0.0; 100]).is_empty());
    }

    #[test]
    fn per_feature_normalisation_gives_zero_mean_unit_unbiased_std() {
        let mut g = Lcg(5);
        let t = 200;
        let mut x: Vec<f32> = (0..t * FEAT_DIM).map(|i| 3.0 * (i % FEAT_DIM) as f32 * 0.01 - 12.0 + g.next_f32()).collect();
        normalize_per_feature(&mut x, t);
        for c in 0..FEAT_DIM {
            let m: f64 = (0..t).map(|r| x[r * FEAT_DIM + c] as f64).sum::<f64>() / t as f64;
            let v: f64 = (0..t).map(|r| (x[r * FEAT_DIM + c] as f64 - m).powi(2)).sum::<f64>() / (t - 1) as f64;
            assert!(m.abs() < 1e-4 && (v.sqrt() - 1.0).abs() < 1e-3, "channel {c}: mean {m} std {}", v.sqrt());
        }
    }

    #[test]
    fn channel_major_layout_is_transposed() {
        let x: Vec<f32> = (0..3 * FEAT_DIM).map(|i| i as f32).collect();
        let ct = to_channel_major(&x, 3);
        assert_eq!(ct[0], 0.0);
        assert_eq!(ct[1], FEAT_DIM as f32); // channel 0, frame 1
        assert_eq!(ct[3], 1.0); // channel 1, frame 0
    }

    // ── Vocab ──

    #[test]
    fn vocab_accepts_a_valid_file_and_finds_blank() {
        let v = Vocab::parse_strict("<unk> 0\n▁a 1\nb 2\n<blk> 3\n").unwrap();
        assert_eq!((v.len(), v.blank_id), (4, 3));
        assert_eq!(v.detokenize(&[1, 2]).0, "ab");
    }

    #[test]
    fn vocab_rejects_every_kind_of_mismatch() {
        let bad = [
            ("<unk> 0\na 1\n<blk> 5\n", "gap"),
            ("<unk> 0\na 1\na 1\n<blk> 3\n", "duplicate id"),
            ("<unk> 0\na 1\nb 2\n", "last token is not <blk>"),
            ("<blk> 0\na 1\nb 2\n", "blank first"),
            ("<unk> 0\n<blk> 1\nb 2\n<blk> 3\n", "two blanks"),
            ("<unk> 0\nfoo bar 1\n<blk> 2\n", "three fields"),
            ("<unk> 0\nfoo\n<blk> 2\n", "one field"),
            ("<unk> 0\na x\n<blk> 2\n", "non numeric id"),
            ("", "empty"),
        ];
        for (text, what) in bad {
            assert!(Vocab::parse_strict(text).is_err(), "must reject: {what}");
        }
    }

    #[test]
    fn detokenize_skips_control_tokens_and_reports_them() {
        let v = Vocab::parse_strict("<unk> 0\n<|ru|> 1\n▁при 2\nвет 3\n<blk> 4\n").unwrap();
        assert_eq!(v.detokenize(&[1, 2, 3, 0]), ("привет".to_string(), 2));
    }

    // ── TDT loop (SYNTHETIC: scripted joiner) ──

    #[test]
    fn tdt_sherpa_rule_zero_duration_moves_one_frame_and_commits_states() {
        let info = tiny_info();
        let fb = Fbank::new(FbankConfig::release());
        // call 1: token 3 dur 0 -> emit, t += 1 ; call 2: blank dur 2 -> t += 2 ; call 3: token 5 dur 1 -> emit, t += 1
        let mut be = Scripted::new(vec![(3, 0), (9, 2), (5, 1)], 4);
        let out = transcribe(&mut be, &info, &fb, TdtRule::Sherpa, &noise(16000, 0.05, 1)).unwrap();
        assert_eq!(out.token_ids, vec![3, 5]);
        assert_eq!(be.joiner_calls, 3, "t: 0 ->1 ->3 ->4 stops");
        assert_eq!(out.stats.zero_skips, 1);
        assert_eq!(out.stats.nonblank_zero_skips, 1);
        // decoder calls: blank (zero state), then token 3 with the state produced by the blank call, then token 5 with the state from token 3
        assert_eq!(be.decoder_calls, vec![(9, 0.0), (3, 1.0), (5, 2.0)]);
        assert_eq!(out.text, "t3t5");
    }

    #[test]
    fn tdt_nemo_rule_zero_duration_token_stays_on_the_frame() {
        let info = tiny_info();
        let fb = Fbank::new(FbankConfig::release());
        // frame 0: token 3 dur 0 (stay) , token 4 dur 1 -> advance ; frame 1: blank dur 0 -> forced to 1
        let mut be = Scripted::new(vec![(3, 0), (4, 1), (9, 0)], 2);
        let out = transcribe(&mut be, &info, &fb, TdtRule::Nemo { max_symbols: 10 }, &noise(16000, 0.05, 1)).unwrap();
        assert_eq!(out.token_ids, vec![3, 4]);
        assert_eq!(be.joiner_calls, 3);
    }

    #[test]
    fn tdt_nemo_rule_is_bounded_by_max_symbols() {
        let info = tiny_info();
        let fb = Fbank::new(FbankConfig::release());
        let mut be = Scripted::new(vec![(3, 0); 50], 1);
        let out = transcribe(&mut be, &info, &fb, TdtRule::Nemo { max_symbols: 3 }, &noise(16000, 0.05, 1)).unwrap();
        assert_eq!(out.token_ids.len(), 3, "3 symbols on the frame, then the frame advances and the loop ends");
    }

    #[test]
    fn transcribe_returns_hard_errors_for_wrong_model_output_sizes() {
        struct BadJoiner;
        impl Backend for BadJoiner {
            fn encode(&mut self, _: &[f32], _: usize) -> Result<Encoded, String> {
                Ok(Encoded { data: vec![0.0; ENC_DIM * 2], t_enc: 2 })
            }
            fn decoder(&mut self, _: i32, _: &[f32], _: &[f32]) -> Result<DecoderOut, String> {
                Ok(DecoderOut { out: vec![0.0; 2], s0: vec![0.0; 2], s1: vec![0.0; 2] })
            }
            fn joiner(&mut self, _: &[f32], _: &[f32]) -> Result<Vec<f32>, String> {
                Ok(vec![0.0; 10]) // not vocab + 5
            }
        }
        let err = transcribe(&mut BadJoiner, &tiny_info(), &Fbank::new(FbankConfig::release()), TdtRule::Sherpa, &noise(16000, 0.05, 1)).unwrap_err();
        assert!(err.contains("logits"), "{err}");
    }

    #[test]
    fn too_short_audio_is_an_empty_result_not_an_error() {
        let mut be = Scripted::new(vec![], 1);
        let out = transcribe(&mut be, &tiny_info(), &Fbank::new(FbankConfig::hann_nopad()), TdtRule::Sherpa, &[0.0; 100]).unwrap();
        assert!(out.text.is_empty() && out.stats.feature_frames == 0);
    }

    // ── Segmenter (SYNTHETIC audio) ──

    fn run_segmenter(audio: &[f32], chunk: usize) -> Vec<SegEvent> {
        let mut s = Segmenter::new(SegmenterConfig::default());
        let mut ev = Vec::new();
        for c in audio.chunks(chunk) {
            ev.extend(s.push(c));
        }
        ev.extend(s.finish());
        ev
    }
    fn segments(ev: &[SegEvent]) -> Vec<(usize, EndReason)> {
        ev.iter().filter_map(|e| if let SegEvent::Segment { samples, reason, .. } = e { Some((samples.len(), *reason)) } else { None }).collect()
    }

    #[test]
    fn segmenter_silence_produces_no_events() {
        assert!(run_segmenter(&vec![0.0; SAMPLE_RATE * 5], 5120).is_empty());
    }

    #[test]
    fn segmenter_one_burst_gives_one_segment_with_preroll_and_pause_tail() {
        let mut a = vec![0.0f32; SAMPLE_RATE];
        a.extend(tone(SAMPLE_RATE, 0.1));
        a.extend(vec![0.0f32; SAMPLE_RATE * 2]);
        let ev = run_segmenter(&a, 5120);
        let segs = segments(&ev);
        assert_eq!(segs.len(), 1, "{ev:?}");
        // 0.2 s preroll + 1 s tone + 0.8 s of silence that ended it
        assert_eq!(segs[0], (SAMPLE_RATE * 2, EndReason::Pause));
        match &ev[0] {
            SegEvent::Started { at_sample, .. } => assert_eq!(*at_sample, (SAMPLE_RATE - SAMPLE_RATE / 5) as u64),
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn segmenter_result_does_not_depend_on_chunk_size() {
        let mut a = noise(SAMPLE_RATE / 2, 0.001, 1);
        a.extend(tone(SAMPLE_RATE * 2, 0.08));
        a.extend(noise(SAMPLE_RATE * 2, 0.001, 2));
        a.extend(tone(SAMPLE_RATE, 0.08));
        let reference = segments(&run_segmenter(&a, 5120));
        for chunk in [1usize, 7, 160, 441, 16000, 100000] {
            assert_eq!(segments(&run_segmenter(&a, chunk)), reference, "chunk {chunk}");
        }
        assert_eq!(reference.len(), 2);
    }

    #[test]
    fn segmenter_long_speech_is_cut_at_max_length_and_the_rest_flushed_on_stop() {
        let ev = run_segmenter(&tone(SAMPLE_RATE * 25, 0.1), 5120);
        let segs = segments(&ev);
        assert_eq!(segs.iter().map(|s| s.1).collect::<Vec<_>>(), vec![EndReason::MaxLength, EndReason::MaxLength, EndReason::Stop]);
        // a steady tone has no dip: the cut lands somewhere in the last 2 s, never beyond the limit
        assert!(segs[0].0 <= SAMPLE_RATE * 10 && segs[0].0 >= SAMPLE_RATE * 8, "{segs:?}");
    }

    /// Concatenating all segments of a continuous utterance must give back exactly the input from the first segment's
    /// start: the quiet-point cut neither loses nor duplicates a sample.
    fn assert_contiguous(audio: &[f32], ev: &[SegEvent]) {
        let mut pos: Option<usize> = None;
        for e in ev {
            if let SegEvent::Segment { samples, start_sample, .. } = e {
                let start = *start_sample as usize;
                if let Some(p) = pos {
                    assert_eq!(start, p, "segment starts at {start}, previous ended at {p}");
                }
                assert_eq!(&audio[start..start + samples.len()], &samples[..], "segment content differs from the input at {start}");
                pos = Some(start + samples.len());
            }
        }
        assert_eq!(pos, Some(audio.len()), "the segments do not reach the end of the input");
    }

    #[test]
    fn segmenter_forced_cut_goes_to_the_quietest_point_and_loses_nothing() {
        // 14 s of loud noise with a 120 ms dip (rms 0.003, below continue) 8.7 s after the start
        let mut audio = noise(SAMPLE_RATE * 14, 0.05, 21);
        let dip = SAMPLE_RATE * 87 / 10;
        for (k, x) in noise(SAMPLE_RATE * 12 / 100, 0.003, 22).into_iter().enumerate() {
            audio[dip + k] = x;
        }
        for chunk in [160usize, 441, 5120] {
            let ev = run_segmenter(&audio, chunk);
            let segs: Vec<(usize, u64, EndReason)> = ev.iter().filter_map(|e| if let SegEvent::Segment { samples, start_sample, reason } = e { Some((samples.len(), *start_sample, *reason)) } else { None }).collect();
            assert_eq!(segs.len(), 2, "chunk {chunk}: {segs:?}");
            assert_eq!((segs[0].2, segs[1].2), (EndReason::MaxLength, EndReason::Stop));
            let cut = segs[0].1 as usize + segs[0].0; // stream position of the cut
            let centre = dip + SAMPLE_RATE * 6 / 100;
            assert!(cut.abs_diff(centre) <= SAMPLE_RATE / 20, "chunk {chunk}: cut at {cut}, dip centre {centre}");
            assert_contiguous(&audio, &ev);
        }
    }

    #[test]
    fn segmenter_forced_cut_without_any_dip_is_still_contiguous_and_bounded() {
        let audio = noise(SAMPLE_RATE * 33, 0.05, 31);
        let ev = run_segmenter(&audio, 5120);
        let segs = segments(&ev);
        assert!(segs.len() >= 4 && segs.iter().all(|s| s.0 <= SAMPLE_RATE * 10), "{segs:?}");
        assert_contiguous(&audio, &ev);
    }

    #[test]
    fn segmenter_forced_cut_carries_the_quiet_tail_into_the_pause_rule() {
        // speech, a 0.3 s dip at ~9.5 s, 0.35 s more speech, then silence: the dip is split between the two segments
        // and the second segment must still end by the pause rule (not wait for a whole new 0.8 s)
        let mut audio = noise(SAMPLE_RATE * 95 / 10, 0.05, 41);
        audio.extend(noise(SAMPLE_RATE * 3 / 10, 0.002, 42));
        audio.extend(noise(SAMPLE_RATE * 35 / 100, 0.05, 43));
        audio.extend(vec![0.0; SAMPLE_RATE * 2]);
        let ev = run_segmenter(&audio, 1600);
        let reasons: Vec<EndReason> = segments(&ev).iter().map(|s| s.1).collect();
        assert_eq!(reasons, vec![EndReason::MaxLength, EndReason::Pause], "{ev:?}");
    }

    #[test]
    fn segmenter_stop_flushes_a_short_open_segment_and_pause_drops_are_explicit() {
        // 0.1 s burst, stream stops right away: flushed (model limit is one 10 ms frame)
        let mut a = tone(SAMPLE_RATE / 10, 0.1);
        let ev = run_segmenter(&a, 5120);
        assert_eq!(segments(&ev).len(), 1);
        assert_eq!(segments(&ev)[0].1, EndReason::Stop);
        // same burst followed by 1 s of silence: ended by the pause, speech part (0.1 s, no preroll) shorter than the
        // 0.3 s minimum -> explicit Dropped event, not a silent drop
        a.extend(vec![0.0; SAMPLE_RATE]);
        let ev = run_segmenter(&a, 5120);
        assert!(segments(&ev).is_empty());
        assert!(ev.iter().any(|e| matches!(e, SegEvent::Dropped { .. })), "{ev:?}");
    }

    #[test]
    fn segmenter_known_limitation_constant_background_above_start_threshold_never_pauses() {
        // A steady hum/noise at RMS 0.01 (-40 dBFS) is above start_rms (0.004): the segmenter never sees silence and
        // can only cut by length. This is why the RMS segmenter is not a VAD (README, "Segmentation").
        let ev = run_segmenter(&noise(SAMPLE_RATE * 25, 0.01, 9), 5120);
        let segs = segments(&ev);
        assert!(!segs.is_empty() && segs.iter().all(|s| s.1 != EndReason::Pause), "{segs:?}");
    }

    #[test]
    fn audio_meter_counts_and_levels() {
        let mut m = AudioMeter::default();
        m.add_i16(&[0, 16384, -16384, i16::MAX]);
        assert_eq!((m.chunks, m.samples, m.clipped), (1, 4, 1));
        assert!((m.peak - 32767.0 / 32768.0).abs() < 1e-6);
        let (n, rms, _) = m.take_window();
        assert_eq!(n, 4);
        assert!(rms > 0.55 && rms < 0.65, "{rms}");
        assert_eq!(m.take_window().0, 0);
    }

    // ── ORT build detection / variant selection (SYNTHETIC) ──

    #[test]
    fn legacy_ort_detection_uses_the_real_build_info_strings() {
        // strings read from the official Linux x64 libraries (GetBuildInfoString)
        let legacy = [
            "ORT Build Info: git-branch=HEAD, git-commit-id=058787ceea, build type=Release", // 1.24.2 (the build NexQ pins)
            "ORT Build Info: git-branch=HEAD, git-commit-id=470ae16099, build type=Release", // 1.24.1
        ];
        let current = [
            "ORT Build Info: git-branch=HEAD, git-commit-id=7a71bc575b, fp8-kv-cache=1, build type=Release", // 1.25.0
            "ORT Build Info: git-branch=HEAD, git-commit-id=8c546c37b4, fp8-kv-cache=1, build type=Release", // 1.26.0
            "ORT Build Info: git-branch=HEAD, git-commit-id=da9b5e364c, fp8-kv-cache=1, build type=Release", // 1.28.0
            "ORT Build Info: git-branch=, git-commit-id=, fp8-kv-cache=1, build type=Release",               // sherpa-onnx's own build
            "",
        ];
        assert!(legacy.iter().all(|s| is_legacy_ort(s)));
        assert!(!current.iter().any(|s| is_legacy_ort(s)));
    }

    #[test]
    fn variant_selection_default_legacy_override_and_bad_names() {
        let (v, log) = select_variant(None, None, false);
        assert_eq!((v.fbank.name, v.tdt), ("release", TdtRule::Sherpa));
        assert!(log.is_empty());
        let (v, log) = select_variant(None, None, true);
        assert_eq!(v.fbank.name, "test_onnx");
        assert!(log.iter().any(|(l, m)| *l == "info" && m.contains("auto-selected")));
        // explicit choice beats the automatic one
        let (v, _) = select_variant(Some("release"), None, true);
        assert_eq!(v.fbank.name, "release");
        // unknown names are reported and ignored; on a legacy build the automatic choice still applies
        let (v, log) = select_variant(Some("nonsense"), Some("nope"), true);
        assert_eq!((v.fbank.name, v.tdt), ("test_onnx", TdtRule::Sherpa));
        assert_eq!(log.iter().filter(|(l, _)| *l == "warn").count(), 2);
        let (v, _) = select_variant(None, Some("nemo"), false);
        assert_eq!(v.tdt, TdtRule::Nemo { max_symbols: 10 });
    }

    // ── driver (SYNTHETIC: scripted backend, scripted audio) ──

    struct StreamRun {
        report: DriverReport,
        /// (number of audio chunks consumed when the result was delivered, result)
        emissions: Vec<(usize, Emission)>,
        logs: Vec<String>,
        total_chunks: usize,
    }

    fn to_i16_chunks(audio: &[f32], chunk: usize) -> Vec<Vec<i16>> {
        audio.chunks(chunk).map(|c| c.iter().map(|&x| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16).collect()).collect()
    }

    /// Feeds `audio` in `chunk`-sample messages, then `Stop`. `fail_emit_at`: delivery of that result number fails.
    fn run_stream<B: Backend>(be: &mut B, info: &ModelInfo, audio: &[f32], chunk: usize, fail_emit_at: Option<usize>) -> Result<StreamRun, String> {
        run_stream_with(be, info, audio, chunk, fail_emit_at, Variant::reference())
    }

    fn run_stream_with<B: Backend>(be: &mut B, info: &ModelInfo, audio: &[f32], chunk: usize, fail_emit_at: Option<usize>, variant: Variant) -> Result<StreamRun, String> {
        let chunks = to_i16_chunks(audio, chunk);
        let total_chunks = chunks.len();
        let queue = std::cell::RefCell::new(VecDeque::from(chunks));
        let consumed = std::cell::Cell::new(0usize);
        let logs = std::cell::RefCell::new(Vec::<String>::new());
        let mut emissions: Vec<(usize, Emission)> = Vec::new();
        let mut opts = DriverOptions::new(variant, 7);
        opts.stats_interval = std::time::Duration::from_secs(3600);
        let mut recv = |_t: std::time::Duration| match queue.borrow_mut().pop_front() {
            Some(c) => {
                consumed.set(consumed.get() + 1);
                RecvEvent::Samples(c)
            }
            None => RecvEvent::Stop,
        };
        let stopped = || false;
        let mut emit = |e: Emission| -> Result<(), String> {
            if fail_emit_at == Some(emissions.len()) {
                return Err("receiver closed".into());
            }
            emissions.push((consumed.get(), e));
            Ok(())
        };
        let debug = |level: &str, msg: &str| logs.borrow_mut().push(format!("[{level}] {msg}"));
        let report = drive(be, info, opts, &mut recv, &stopped, &mut emit, &debug)?;
        drop(emit);
        Ok(StreamRun { report, emissions, logs: logs.into_inner(), total_chunks })
    }

    fn loud(secs: f32, seed: u64) -> Vec<f32> {
        noise((SAMPLE_RATE as f32 * secs) as usize, 0.05, seed)
    }
    fn quiet(secs: f32, seed: u64) -> Vec<f32> {
        noise((SAMPLE_RATE as f32 * secs) as usize, 0.0008, seed)
    }
    fn pairs() -> Vec<(usize, usize)> {
        [(3, 1), (9, 4)].repeat(20) // every 2-frame segment: token "t3", then blank + skip 4
    }

    #[test]
    fn stop_flag_does_not_discard_audio_that_is_already_queued() {
        // The application sets the stop flag first and queues `Stop` behind the audio that was already sent; here
        // there is even no `Stop` (an empty queue answers `Timeout`). Everything queued must still be recognised.
        let audio = [quiet(0.5, 1), loud(1.0, 2), quiet(1.2, 3), loud(0.6, 4)].concat();
        for with_stop_message in [false, true] {
            let chunks = to_i16_chunks(&audio, 441);
            let total = chunks.len();
            let queue = std::cell::RefCell::new(VecDeque::from(chunks));
            let consumed = std::cell::Cell::new(0usize);
            let logs = std::cell::RefCell::new(Vec::<String>::new());
            let mut texts: Vec<String> = vec![];
            let mut opts = DriverOptions::new(Variant::reference(), 1);
            opts.stats_interval = std::time::Duration::from_secs(3600);
            let mut recv = |_t: std::time::Duration| match queue.borrow_mut().pop_front() {
                Some(c) => {
                    consumed.set(consumed.get() + 1);
                    RecvEvent::Samples(c)
                }
                None if with_stop_message => RecvEvent::Stop,
                None => RecvEvent::Timeout,
            };
            let stopped = || consumed.get() >= 10; // the user pressed Stop while ~90% of the audio was still queued
            let mut emit = |e: Emission| -> Result<(), String> {
                texts.push(e.text);
                Ok(())
            };
            let debug = |level: &str, msg: &str| logs.borrow_mut().push(format!("[{level}] {msg}"));
            let mut be = Scripted::new(pairs(), 2);
            let report = drive(&mut be, &tiny_info(), opts, &mut recv, &stopped, &mut emit, &debug).unwrap();
            drop(emit);
            assert_eq!(consumed.get(), total, "all queued chunks must be consumed (stop message: {with_stop_message})");
            assert_eq!(texts, vec!["t3", "t3"], "both phrases, including the cut-off last one (stop message: {with_stop_message})");
            assert_eq!(report.samples as usize, audio.len());
            assert!(logs.borrow().iter().any(|l| l.contains("stop requested")), "{:?}", logs.borrow());
        }
    }

    #[test]
    fn driver_emits_after_pause_and_flushes_at_stop() {
        // quiet, phrase, 1.2 s pause, phrase WITHOUT trailing silence, then Stop.
        let audio = [quiet(0.5, 1), loud(1.0, 2), quiet(1.2, 3), loud(0.6, 4)].concat();
        let mut texts: Vec<Vec<String>> = vec![];
        for chunk in [160usize, 441, 5120] {
            let mut be = Scripted::new(pairs(), 2);
            let run = run_stream(&mut be, &tiny_info(), &audio, chunk, None).unwrap();
            assert_eq!(run.emissions.len(), 2, "chunk {chunk}: {:?}", run.logs);
            let (c0, e0) = &run.emissions[0];
            let (c1, e1) = &run.emissions[1];
            assert!(*c0 < run.total_chunks, "first phrase must be delivered after the pause, before Stop (chunk {chunk})");
            assert_eq!(*c1, run.total_chunks, "second phrase can only be delivered by the flush at Stop (chunk {chunk})");
            assert_eq!((e0.segment_id.as_str(), e1.segment_id.as_str()), ("ort_7", "ort_8"));
            assert_eq!((e0.text.as_str(), e1.text.as_str()), ("t3", "t3"));
            assert!(e0.duration_ms > 1500 && e0.duration_ms < 2500, "{}", e0.duration_ms);
            assert!(e0.timestamp_ms > 0 && e0.timestamp_ms + e0.duration_ms <= epoch_ms() + 50);
            assert!(e1.timestamp_ms >= e0.timestamp_ms);
            let r = &run.report;
            assert_eq!((r.segments_started, r.segments_decoded, r.results_sent, r.send_errors, r.segments_failed), (2, 2, 2, 0, 0));
            assert_eq!(r.samples as usize, audio.len());
            texts.push(run.emissions.iter().map(|(_, e)| e.text.clone()).collect());
        }
        assert!(texts.windows(2).all(|w| w[0] == w[1]), "the result must not depend on the sender's chunk size");
    }

    #[test]
    fn driver_logs_every_stage() {
        let audio = [quiet(0.5, 1), loud(1.0, 2), quiet(1.2, 3), loud(0.6, 4)].concat();
        let mut be = Scripted::new(pairs(), 2);
        let run = run_stream(&mut be, &tiny_info(), &audio, 320, None).unwrap();
        let all = run.logs.join("\n");
        for needle in [
            "frontend:", "decoder loop:", "segmenter (RMS", "self-test OK", "ready; waiting for audio", "first audio chunk reached",
            "segment started", "ended (Pause)", "ended (Stop)", "feature frames -> 2 encoder frames", "TranscriptResult sent", "NeMo stream ended",
        ] {
            assert!(all.contains(needle), "log lacks {needle:?}:\n{all}");
        }
    }

    #[test]
    fn driver_stops_when_delivery_fails_and_says_so() {
        let audio = [quiet(0.5, 1), loud(1.0, 2), quiet(1.2, 3), loud(1.0, 4), quiet(1.2, 5)].concat();
        let mut be = Scripted::new(pairs(), 2);
        let run = run_stream(&mut be, &tiny_info(), &audio, 320, Some(0)).unwrap();
        assert_eq!((run.report.send_errors, run.report.results_sent), (1, 0));
        assert!(run.emissions.is_empty());
        assert!(run.logs.iter().any(|l| l.starts_with("[error]") && l.contains("NOT delivered")), "{:?}", run.logs);
        // it stopped reading at the failed delivery instead of decoding the rest
        assert_eq!(run.report.segments_decoded, 1);
    }

    struct FailNth {
        inner: Scripted,
        encode_calls: usize,
        fail_on: usize,
    }
    impl Backend for FailNth {
        fn encode(&mut self, f: &[f32], n: usize) -> Result<Encoded, String> {
            self.encode_calls += 1;
            if self.encode_calls == self.fail_on {
                return Err("synthetic encoder failure".into());
            }
            self.inner.encode(f, n)
        }
        fn decoder(&mut self, t: i32, a: &[f32], b: &[f32]) -> Result<DecoderOut, String> {
            self.inner.decoder(t, a, b)
        }
        fn joiner(&mut self, e: &[f32], d: &[f32]) -> Result<Vec<f32>, String> {
            self.inner.joiner(e, d)
        }
    }

    #[test]
    fn driver_survives_a_failed_segment_and_logs_the_error() {
        let audio = [quiet(0.5, 1), loud(1.0, 2), quiet(1.2, 3), loud(1.0, 4), quiet(1.2, 5)].concat();
        // encode call 1 = start-up self-test, 2 = first phrase (fails), 3 = second phrase
        let mut be = FailNth { inner: Scripted::new(pairs(), 2), encode_calls: 0, fail_on: 2 };
        let run = run_stream(&mut be, &tiny_info(), &audio, 320, None).unwrap();
        assert_eq!((run.report.segments_failed, run.report.results_sent), (1, 1), "{:?}", run.logs);
        assert!(run.logs.iter().any(|l| l.starts_with("[error]") && l.contains("inference failed") && l.contains("synthetic encoder failure")));
    }

    #[test]
    fn driver_self_test_failure_is_a_hard_error() {
        let mut be = FailNth { inner: Scripted::new(pairs(), 2), encode_calls: 0, fail_on: 1 };
        let err = run_stream(&mut be, &tiny_info(), &quiet(1.0, 1), 320, None).err().expect("must fail");
        assert!(err.contains("self-test failed") && err.contains("synthetic encoder failure"), "{err}");
    }

    #[test]
    fn driver_reports_empty_results_and_dropped_segments_instead_of_hiding_them() {
        // blank only: the model returns no text
        let mut be = Scripted::new([(9, 4)].repeat(40), 2);
        let audio = [quiet(0.5, 1), loud(1.0, 2), quiet(1.2, 3)].concat();
        let run = run_stream(&mut be, &tiny_info(), &audio, 320, None).unwrap();
        assert_eq!((run.report.segments_decoded, run.report.segments_empty, run.report.results_sent), (1, 1, 0));
        assert!(run.logs.iter().any(|l| l.contains("model returned NO text")));
        // a 50 ms click is shorter than the minimum phrase length (the 200 ms preroll counts towards it, so a 100 ms
        // click would still be sent to the model - deliberate: a short "Да?" must not be lost)
        let mut be = Scripted::new(pairs(), 2);
        let audio = [quiet(0.5, 1), loud(0.05, 2), quiet(1.2, 3)].concat();
        let run = run_stream(&mut be, &tiny_info(), &audio, 320, None).unwrap();
        assert_eq!((run.report.segments_dropped, run.report.results_sent), (1, 0));
        assert!(run.logs.iter().any(|l| l.contains("segment dropped")));
    }

    // ═════════════════════════ REAL-DATA TESTS (ignored by default) ═════════════════════════

    fn env_path(name: &str) -> PathBuf {
        PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
    }

    /// Reference process: `ort_ab serve` (tools/ab/ort_ab.cpp) — ONNX Runtime 1.24 with the real int8 model.
    struct SubprocessBackend {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
        pred_hidden: usize,
        pred_layers: usize,
    }
    impl SubprocessBackend {
        fn start(ort_ab: &Path, model_dir: &Path) -> (Self, ModelInfo) {
            let mut child = Command::new(ort_ab)
                .args(["serve", "--model-dir"])
                .arg(model_dir)
                .args(["--threads", "2"])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .expect("spawn ort_ab");
            let stdin = child.stdin.take().unwrap();
            let mut stdout = BufReader::new(child.stdout.take().unwrap());
            let mut line = String::new();
            stdout.read_line(&mut line).unwrap();
            let f: Vec<&str> = line.split_whitespace().collect();
            let (layers, hidden): (usize, usize) = (f[1].parse().unwrap(), f[2].parse().unwrap());
            let vocab = Vocab::parse_strict(&std::fs::read_to_string(model_dir.join("tokens.txt")).unwrap()).unwrap();
            assert_eq!(vocab.len(), f[3].parse::<usize>().unwrap());
            assert_eq!(vocab.blank_id, f[4].parse::<usize>().unwrap());
            let info = ModelInfo { vocab, pred_layers: layers, pred_hidden: hidden, per_feature_norm: f[0] == "per_feature" };
            (Self { child, stdin, stdout, pred_hidden: hidden, pred_layers: layers }, info)
        }
        fn put_f32(&mut self, v: &[f32]) {
            let mut b = Vec::with_capacity(v.len() * 4);
            for x in v {
                b.extend_from_slice(&x.to_le_bytes());
            }
            self.stdin.write_all(&b).unwrap();
        }
        fn get_f32(&mut self, n: usize) -> Vec<f32> {
            let mut b = vec![0u8; n * 4];
            self.stdout.read_exact(&mut b).unwrap();
            b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
        }
        fn get_u32(&mut self) -> u32 {
            let mut b = [0u8; 4];
            self.stdout.read_exact(&mut b).unwrap();
            u32::from_le_bytes(b)
        }
    }
    impl Drop for SubprocessBackend {
        fn drop(&mut self) {
            let _ = self.stdin.write_all(b"Q");
            let _ = self.child.wait();
        }
    }
    impl Backend for SubprocessBackend {
        fn encode(&mut self, feats_ct: &[f32], frames: usize) -> Result<Encoded, String> {
            self.stdin.write_all(b"E").unwrap();
            self.stdin.write_all(&(frames as u32).to_le_bytes()).unwrap();
            self.put_f32(feats_ct);
            self.stdin.flush().unwrap();
            let t = self.get_u32() as usize;
            Ok(Encoded { data: self.get_f32(ENC_DIM * t), t_enc: t })
        }
        fn decoder(&mut self, token: i32, s0: &[f32], s1: &[f32]) -> Result<DecoderOut, String> {
            self.stdin.write_all(b"D").unwrap();
            self.stdin.write_all(&token.to_le_bytes()).unwrap();
            self.put_f32(s0);
            self.put_f32(s1);
            self.stdin.flush().unwrap();
            let sn = self.pred_layers * self.pred_hidden;
            Ok(DecoderOut { out: self.get_f32(self.pred_hidden), s0: self.get_f32(sn), s1: self.get_f32(sn) })
        }
        fn joiner(&mut self, e: &[f32], d: &[f32]) -> Result<Vec<f32>, String> {
            self.stdin.write_all(b"J").unwrap();
            self.put_f32(e);
            self.put_f32(d);
            self.stdin.flush().unwrap();
            let n = self.get_u32() as usize;
            Ok(self.get_f32(n))
        }
    }

    fn expected_texts() -> HashMap<String, String> {
        let mut m = HashMap::new();
        if let Ok(p) = std::env::var("NEXQ_EXPECT_FILE") {
            for l in std::fs::read_to_string(p).unwrap().lines() {
                if let Some((k, v)) = l.split_once('|') {
                    m.insert(k.trim().to_string(), v.trim().to_string());
                }
            }
        }
        m
    }

    fn wav_files(dir: &Path) -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().map_or(false, |x| x == "wav")).collect();
        v.sort();
        v
    }

    /// Rust Fbank vs the real kaldi-native-fbank dumps produced by `tools/ab/knf_dump` (names `<stem>.<config>.raw.f32`,
    /// config names as in run_ab.py: testonnx, release, master, hann_nopad).
    /// NEXQ_WAV_DIR=<16 kHz wavs>  NEXQ_KNF_DIR=<dump dir>
    #[test]
    #[ignore = "needs WAVs and real knf dumps (see README)"]
    fn real_knf_feature_comparison() {
        let (wavs, dumps) = (env_path("NEXQ_WAV_DIR"), env_path("NEXQ_KNF_DIR"));
        let mut worst_mean = 0.0f64;
        let mut compared = 0;
        for wav in wav_files(&wavs) {
            let stem = wav.file_stem().unwrap().to_string_lossy().to_string();
            let audio = read_wav(&wav);
            for (name, cfg) in [("testonnx", FbankConfig::test_onnx()), ("release", FbankConfig::release()), ("master", FbankConfig::master()), ("hann_nopad", FbankConfig::hann_nopad())] {
                let path = dumps.join(format!("{stem}.{name}.raw.f32"));
                let Ok(bytes) = std::fs::read(&path) else { continue };
                let reference: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                let mut ours = Fbank::new(cfg).compute(&audio);
                assert_eq!(ours.len(), reference.len(), "{stem}/{name}: frame count differs from real knf ({} vs {} floats)", ours.len(), reference.len());
                let frames = ours.len() / FEAT_DIM;
                let (mut max_abs, mut sum_abs) = (0.0f64, 0.0f64);
                for (a, b) in ours.iter().zip(&reference) {
                    let d = (a - b).abs() as f64;
                    max_abs = max_abs.max(d);
                    sum_abs += d;
                }
                let mean_abs = sum_abs / ours.len() as f64;
                // after normalisation (what the model sees)
                let mut r = reference.clone();
                normalize_per_feature(&mut ours, frames);
                normalize_per_feature(&mut r, frames);
                let max_norm = ours.iter().zip(&r).map(|(a, b)| (a - b).abs() as f64).fold(0.0, f64::max);
                println!("{stem:14} {name:11} frames={frames:5} raw: max|d|={max_abs:.5} mean|d|={mean_abs:.7}   normalised: max|d|={max_norm:.5}");
                worst_mean = worst_mean.max(mean_abs);
                compared += 1;
                assert!(mean_abs < 1e-3 && max_norm < 0.05, "{stem}/{name}: Rust Fbank deviates from real knf");
            }
        }
        assert!(compared > 0, "no dumps found in {}", dumps.display());
        println!("compared {compared} (wav, config) pairs; worst mean|d| = {worst_mean:.7}");
    }

    /// Rust front end + Rust TDT loop + REAL model (reference process) on real WAVs.
    /// NEXQ_ORT_AB=<ort_ab binary> NEXQ_MODEL_DIR=<model dir> NEXQ_WAV_DIR=<wavs> [NEXQ_EXPECT_FILE=<stem|text lines>]
    /// [NEXQ_VARIANTS=release,test_onnx,master,hann_nopad] [NEXQ_TDT=sherpa,nemo]
    #[test]
    #[ignore = "needs the real model and WAVs (see README)"]
    fn real_model_transcription() {
        let (mut be, info) = SubprocessBackend::start(&env_path("NEXQ_ORT_AB"), &env_path("NEXQ_MODEL_DIR"));
        let wavs = wav_files(&env_path("NEXQ_WAV_DIR"));
        let expected = expected_texts();
        let variants = std::env::var("NEXQ_VARIANTS").unwrap_or_else(|_| "release".into());
        let tdts = std::env::var("NEXQ_TDT").unwrap_or_else(|_| "sherpa".into());
        let (mut sum_cer, mut n_cer) = (0.0f32, 0);
        let min_agreement = std::env::var("NEXQ_MAX_CER").ok().and_then(|v| v.parse::<f32>().ok());
        for v in variants.split(',') {
            let cfg = FbankConfig::from_name(v).unwrap_or_else(|| panic!("unknown variant {v}"));
            let fb = Fbank::new(cfg.clone());
            for t in tdts.split(',') {
                let tdt = TdtRule::from_name(t).unwrap();
                let (mut vs, mut vn) = (0.0f32, 0);
                for w in &wavs {
                    let stem = w.file_stem().unwrap().to_string_lossy().to_string();
                    let audio = read_wav(w);
                    let out = transcribe(&mut be, &info, &fb, tdt, &audio).unwrap();
                    let c = expected.get(&stem).map(|e| cer(&out.text, e));
                    if let Some(c) = c {
                        vs += c;
                        vn += 1;
                    }
                    println!(
                        "{:11} {:6} {stem:14} frames={:5} enc={:4} tokens={:3} {} | {}",
                        cfg.name, t, out.stats.feature_frames, out.stats.encoder_frames, out.stats.tokens,
                        c.map_or(String::new(), |c| format!("CER={:5.1}%", c * 100.0)), out.text
                    );
                }
                if vn > 0 {
                    println!("AVG {:11} {:6} CER={:.1}% (n={vn})", cfg.name, t, vs / vn as f32 * 100.0);
                    if cfg.name == "release" && t == "sherpa" {
                        sum_cer += vs;
                        n_cer += vn;
                    }
                }
            }
        }
        if let (Some(max), true) = (min_agreement, n_cer > 0) {
            let avg = sum_cer / n_cer as f32;
            assert!(avg <= max, "average CER {avg} exceeds {max}");
        }
    }

    /// Scenarios built from real speech: segmenter + front end + model. Prints what happened for each scenario.
    /// Same env as real_model_transcription plus NEXQ_SCENARIO_WAV=<path of a speech wav> (default: first wav in the dir).
    #[test]
    #[ignore = "needs the real model and WAVs (see README)"]
    fn real_segmenter_scenarios() {
        let (mut be, info) = SubprocessBackend::start(&env_path("NEXQ_ORT_AB"), &env_path("NEXQ_MODEL_DIR"));
        let dir = env_path("NEXQ_WAV_DIR");
        let speech_path = std::env::var("NEXQ_SCENARIO_WAV").map(PathBuf::from).unwrap_or_else(|_| wav_files(&dir)[0].clone());
        let speech = read_wav(&speech_path);
        let fb = Fbank::new(FbankConfig::release());
        let speech_rms = (speech.iter().map(|x| x * x).sum::<f32>() / speech.len() as f32).sqrt();
        println!("speech: {} ({:.1}s, rms {:.4} = {:.1} dBFS)", speech_path.display(), speech.len() as f32 / 16000.0, speech_rms, dbfs(speech_rms));

        let scale = |g: f32| -> Vec<f32> { speech.iter().map(|x| x * g).collect() };
        let mix = |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().zip(b.iter().chain(std::iter::repeat(&0.0))).map(|(x, y)| x + y).collect() };
        let quiet = |s: usize, rms: f32, seed: u64| noise(SAMPLE_RATE * s, rms, seed);
        let cat = |parts: Vec<Vec<f32>>| -> Vec<f32> { parts.concat() };

        let scenarios: Vec<(&str, Vec<f32>)> = vec![
            ("clean: 2s quiet + speech + 2s quiet", cat(vec![quiet(2, 0.0008, 1), speech.clone(), quiet(2, 0.0008, 2)])),
            ("weak speech (-20 dB)", cat(vec![quiet(2, 0.0008, 3), scale(0.1), quiet(2, 0.0008, 4)])),
            ("very weak speech (-30 dB)", cat(vec![quiet(2, 0.0008, 5), scale(0.0316), quiet(2, 0.0008, 6)])),
            ("noise-only (rms 0.003, below start)", quiet(8, 0.003, 7)),
            ("noise-only (rms 0.02, above start)", quiet(8, 0.02, 8)),
            ("speech over constant hum rms 0.01", mix(&cat(vec![quiet(2, 0.0, 9), speech.clone(), quiet(2, 0.0, 10)]), &noise(SAMPLE_RATE * 15, 0.01, 11))),
            ("speech cut off at stop (no trailing silence)", cat(vec![quiet(1, 0.0008, 12), speech.clone()])),
            ("two phrases 1.2 s apart", cat(vec![quiet(1, 0.0008, 13), speech.clone(), quiet(1, 0.0008, 14), speech.clone(), quiet(2, 0.0008, 15)])),
        ];
        for (name, audio) in scenarios {
            let mut seg = Segmenter::new(SegmenterConfig::default());
            let mut events = Vec::new();
            for c in audio.chunks(5120) {
                events.extend(seg.push(c));
            }
            events.extend(seg.finish());
            println!("── {name}  ({:.1}s)", audio.len() as f32 / 16000.0);
            let mut n_seg = 0;
            for e in &events {
                match e {
                    SegEvent::Started { at_sample, rms } => println!("   started @{:.2}s rms={:.4}", *at_sample as f32 / 16000.0, rms),
                    SegEvent::Dropped { len, reason } => println!("   DROPPED {:.2}s: {reason}", *len as f32 / 16000.0),
                    SegEvent::Segment { samples, start_sample, reason } => {
                        n_seg += 1;
                        let d = transcribe(&mut be, &info, &fb, TdtRule::Sherpa, samples).unwrap();
                        println!("   segment @{:.2}s len={:.2}s end={:?} tokens={} text={:?}", *start_sample as f32 / 16000.0, samples.len() as f32 / 16000.0, reason, d.stats.tokens, d.text);
                    }
                }
            }
            if n_seg == 0 {
                println!("   (no segment reached the model)");
            }
        }
    }

    /// The whole loop (meter → segmenter → front end → model → delivery) on a real stream: all WAVs of the directory one
    /// after another, 1.2 s of faint noise between them, the last one cut off without trailing silence, fed in chunks of
    /// several sizes. Prints every delivery with the number of chunks consumed at that moment.
    /// NEXQ_ORT_AB, NEXQ_MODEL_DIR, NEXQ_WAV_DIR, NEXQ_EXPECT_FILE [, NEXQ_MAX_CER (default 0.20), NEXQ_VARIANT, NEXQ_LEGACY_ORT]
    #[test]
    #[ignore = "needs the real model and WAVs (see README)"]
    fn real_driver_stream() {
        let (mut be, info) = SubprocessBackend::start(&env_path("NEXQ_ORT_AB"), &env_path("NEXQ_MODEL_DIR"));
        let expected = expected_texts();
        let wavs: Vec<PathBuf> = wav_files(&env_path("NEXQ_WAV_DIR")).into_iter().filter(|w| expected.contains_key(&w.file_stem().unwrap().to_string_lossy().to_string())).collect();
        assert!(!wavs.is_empty(), "no WAV has an expected text in NEXQ_EXPECT_FILE");
        let mut audio = quiet(1.0, 100);
        let mut reference = Vec::new();
        for (i, w) in wavs.iter().enumerate() {
            audio.extend(read_wav(w));
            reference.push(expected[&w.file_stem().unwrap().to_string_lossy().to_string()].clone());
            if i + 1 < wavs.len() {
                audio.extend(quiet(1.2, 200 + i as u64));
            }
        }
        let reference = reference.join(" ");
        let max_cer: f32 = std::env::var("NEXQ_MAX_CER").ok().and_then(|v| v.parse().ok()).unwrap_or(0.20);
        // NEXQ_VARIANT=<release|test_onnx|...>; NEXQ_LEGACY_ORT=1 reproduces the app's automatic choice on ORT 1.24.x
        let (variant, _) = select_variant(std::env::var("NEXQ_VARIANT").ok().as_deref(), None, std::env::var("NEXQ_LEGACY_ORT").is_ok());
        println!("front end: {}", variant.fbank.describe());
        println!("stream: {} phrases, {:.1}s", wavs.len(), audio.len() as f32 / 16000.0);
        let mut joined_per_chunk: Vec<String> = vec![];
        for chunk in [320usize, 441, 4096] {
            let run = run_stream_with(&mut be, &info, &audio, chunk, None, variant.clone()).unwrap();
            println!("── chunk {chunk}: {} chunks, report {:?}", run.total_chunks, run.report);
            for (c, e) in &run.emissions {
                println!("   after chunk {c:5}/{} {:6.2}s {} | {}", run.total_chunks, e.duration_ms as f32 / 1000.0, e.segment_id, e.text);
            }
            let print_all = std::env::var("NEXQ_PRINT_LOGS").is_ok();
            for l in run.logs.iter().filter(|l| print_all || l.contains("dropped") || l.contains("NO text") || l.starts_with("[error]") || l.starts_with("[warn]")) {
                println!("   LOG {l}");
            }
            assert_eq!(run.report.send_errors + run.report.segments_failed, 0, "{:?}", run.logs);
            let (last_c, _) = run.emissions.last().expect("nothing was delivered");
            assert_eq!(*last_c, run.total_chunks, "the cut-off last phrase must be delivered by the flush at Stop");
            assert!(run.emissions.iter().rev().skip(1).all(|(c, _)| *c < run.total_chunks), "earlier phrases must be delivered after their pause, not at Stop");
            let joined = run.emissions.iter().map(|(_, e)| e.text.clone()).collect::<Vec<_>>().join(" ");
            let c = cer(&joined, &reference);
            println!("   stream CER = {:.1}%  (reference: {} phrases, delivered: {} results)", c * 100.0, wavs.len(), run.emissions.len());
            assert!(c <= max_cer, "stream CER {c} > {max_cer}");
            joined_per_chunk.push(joined);
        }
        assert!(joined_per_chunk.windows(2).all(|w| w[0] == w[1]), "the transcript must not depend on the sender's chunk size:\n{joined_per_chunk:#?}");
    }
}
