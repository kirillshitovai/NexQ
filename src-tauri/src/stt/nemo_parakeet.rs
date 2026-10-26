//! NeMo Parakeet TDT 0.6B v3 for NexQ 2.26.0 — ONNX Runtime glue and driver loop (revision 3).
//!
//! All numerical logic (Fbank, normalisation, TDT loop, tokens.txt validation, segmenter) lives in
//! `nemo_parakeet_core.rs`, which has no dependencies and was verified against the real model; this file only
//!   * validates the three ONNX sessions against the model contract and `tokens.txt` (hard errors),
//!   * implements `pk::Backend` on top of `ort` (the tensor calls are the ones that compiled in revision 1),
//!   * runs the receive → segment → decode → `TranscriptResult` loop and logs every step to the dev log.
//!
//! Environment switches (read once at stream start, all optional):
//!   NEXQ_PARAKEET_VARIANT   release | test_onnx | master | hann_nopad              — Fbank configuration.
//!                           Default: `release` (sherpa-onnx production recipe, best on ONNX Runtime >= 1.25) or, when
//!                           the loaded library is a known official 1.24.x build, `test_onnx` (see `is_legacy_ort`)
//!   NEXQ_PARAKEET_TDT       sherpa (default) | nemo                              — TDT zero-duration rule
//!   NEXQ_PARAKEET_DUMP_DIR  directory; every segment sent to the model is also written there as 16 kHz WAV
use super::nemo_parakeet_core as pk;
use super::{load_ort_session, AudioMessage};
use pk::{Backend, DecoderOut, Encoded, ModelInfo, Variant, Vocab};
use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use ort::value::Tensor;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

use crate::stt::provider::TranscriptResult;

type DebugLog<'a> = dyn Fn(&str, &str) + 'a;

// ─────────────────────────────────────────────────────────── model contract checks ──────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dt {
    F32,
    I32,
    I64,
    Other,
}

/// (element type, shape) of a tensor input/output, `None` for non-tensors.
fn tensor_sig(dtype: &ort::value::ValueType) -> Option<(Dt, Vec<i64>)> {
    if let ort::value::ValueType::Tensor { ty, shape, .. } = dtype {
        let dt = if *ty == ort::value::TensorElementType::Float32 {
            Dt::F32
        } else if *ty == ort::value::TensorElementType::Int32 {
            Dt::I32
        } else if *ty == ort::value::TensorElementType::Int64 {
            Dt::I64
        } else {
            Dt::Other
        };
        Some((dt, shape.to_vec()))
    } else {
        None
    }
}

fn describe_session(label: &str, s: &Session) -> String {
    let fmt = |name: &str, dtype: &ort::value::ValueType| match tensor_sig(dtype) {
        Some((dt, shape)) => format!("{name} {dt:?}{shape:?}"),
        None => format!("{name} <non-tensor>"),
    };
    let ins: Vec<String> = s.inputs().iter().map(|i| fmt(i.name(), i.dtype())).collect();
    let outs: Vec<String> = s.outputs().iter().map(|o| fmt(o.name(), o.dtype())).collect();
    format!("{label} loaded: inputs=[{}] outputs=[{}]", ins.join(", "), outs.join(", "))
}

fn check_input(s: &Session, idx: usize, who: &str, want: Dt, rank: usize) -> Result<Vec<i64>, String> {
    let input = s.inputs().get(idx).ok_or_else(|| format!("{who}: input #{idx} is missing"))?;
    let (dt, shape) = tensor_sig(input.dtype()).ok_or_else(|| format!("{who}: input #{idx} ({}) is not a tensor", input.name()))?;
    if dt != want || shape.len() != rank {
        return Err(format!("{who}: input #{idx} ({}) is {dt:?}{shape:?}, expected {want:?} of rank {rank}", input.name()));
    }
    Ok(shape)
}

fn check_output(s: &Session, idx: usize, who: &str, want: Dt, rank: Option<usize>) -> Result<Vec<i64>, String> {
    let output = s.outputs().get(idx).ok_or_else(|| format!("{who}: output #{idx} is missing"))?;
    let (dt, shape) = tensor_sig(output.dtype()).ok_or_else(|| format!("{who}: output #{idx} ({}) is not a tensor", output.name()))?;
    if dt != want || rank.map_or(false, |r| shape.len() != r) {
        return Err(format!("{who}: output #{idx} ({}) is {dt:?}{shape:?}, expected {want:?}{}", output.name(), rank.map_or(String::new(), |r| format!(" of rank {r}"))));
    }
    Ok(shape)
}

/// A dimension that is static in the model must match; dynamic dimensions (`-1`) are accepted.
fn expect_dim(who: &str, what: &str, shape: &[i64], axis: usize, want: usize) -> Result<(), String> {
    match shape.get(axis) {
        Some(&d) if d > 0 && d as usize != want => Err(format!("{who}: {what} has dimension {d} on axis {axis}, expected {want} (shape {shape:?})")),
        Some(_) => Ok(()),
        None => Err(format!("{who}: {what} has no axis {axis} (shape {shape:?})")),
    }
}

// ─────────────────────────────────────────────────────────── ORT backend ─────────────────────────

struct OrtBackend {
    encoder: Session,
    decoder: Session,
    joiner: Session,
    pred_layers: usize,
    pred_hidden: usize,
}

impl Backend for OrtBackend {
    fn encode(&mut self, feats_ct: &[f32], frames: usize) -> Result<Encoded, String> {
        let signal = Tensor::from_array(
            Array3::from_shape_vec((1, pk::FEAT_DIM, frames), feats_ct.to_vec()).map_err(|e| format!("NeMo feature shape: {e}"))?,
        )
        .map_err(|e| format!("NeMo audio tensor: {e}"))?;
        let length = Tensor::from_array(Array1::from_vec(vec![frames as i64])).map_err(|e| format!("NeMo length tensor: {e}"))?;
        let outputs = self.encoder.run(ort::inputs![signal, length]).map_err(|e| format!("NeMo encoder: {e}"))?;
        let (shape, enc_data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| format!("NeMo encoder output: {e}"))?;
        if shape.len() != 3 || shape[0] != 1 || shape[1] != pk::ENC_DIM as i64 {
            return Err(format!("NeMo encoder expected [1,{},T], got {shape:?}", pk::ENC_DIM));
        }
        let t_enc = shape[2] as usize;
        if enc_data.len() != pk::ENC_DIM * t_enc {
            return Err(format!("NeMo encoder data length mismatch: {} for T={t_enc}", enc_data.len()));
        }
        // Own the data before the session outputs (which borrow the encoder) are dropped.
        Ok(Encoded { data: enc_data.to_vec(), t_enc })
    }

    fn decoder(&mut self, token: i32, s0: &[f32], s1: &[f32]) -> Result<DecoderOut, String> {
        let target = Tensor::from_array(Array2::from_shape_vec((1, 1), vec![token]).map_err(|e| format!("NeMo target shape: {e}"))?)
            .map_err(|e| format!("NeMo target tensor: {e}"))?;
        let target_len = Tensor::from_array(Array1::from_vec(vec![1i32])).map_err(|e| format!("NeMo target length: {e}"))?;
        let st0 = Tensor::from_array(
            Array3::from_shape_vec((self.pred_layers, 1, self.pred_hidden), s0.to_vec()).map_err(|e| format!("NeMo state0 shape: {e}"))?,
        )
        .map_err(|e| format!("NeMo state0 tensor: {e}"))?;
        let st1 = Tensor::from_array(
            Array3::from_shape_vec((self.pred_layers, 1, self.pred_hidden), s1.to_vec()).map_err(|e| format!("NeMo state1 shape: {e}"))?,
        )
        .map_err(|e| format!("NeMo state1 tensor: {e}"))?;
        let outs = self.decoder.run(ort::inputs![target, target_len, st0, st1]).map_err(|e| format!("NeMo decoder: {e}"))?;
        let (_, dec) = outs[0].try_extract_tensor::<f32>().map_err(|e| format!("NeMo decoder output: {e}"))?;
        let (_, n0) = outs[2].try_extract_tensor::<f32>().map_err(|e| format!("NeMo decoder state0: {e}"))?;
        let (_, n1) = outs[3].try_extract_tensor::<f32>().map_err(|e| format!("NeMo decoder state1: {e}"))?;
        Ok(DecoderOut { out: dec.to_vec(), s0: n0.to_vec(), s1: n1.to_vec() })
    }

    fn joiner(&mut self, enc_frame: &[f32], dec_out: &[f32]) -> Result<Vec<f32>, String> {
        // NeMo joiner inputs are [1, dim, 1], NOT [1, 1, dim].
        let enc = Tensor::from_array(
            Array3::from_shape_vec((1, pk::ENC_DIM, 1), enc_frame.to_vec()).map_err(|e| format!("NeMo joiner encoder shape: {e}"))?,
        )
        .map_err(|e| format!("NeMo joiner encoder tensor: {e}"))?;
        let dec = Tensor::from_array(
            Array3::from_shape_vec((1, self.pred_hidden, 1), dec_out.to_vec()).map_err(|e| format!("NeMo joiner decoder shape: {e}"))?,
        )
        .map_err(|e| format!("NeMo joiner decoder tensor: {e}"))?;
        let jo = self.joiner.run(ort::inputs![enc, dec]).map_err(|e| format!("NeMo joiner: {e}"))?;
        let (_, logits) = jo[0].try_extract_tensor::<f32>().map_err(|e| format!("NeMo logits: {e}"))?;
        Ok(logits.to_vec())
    }
}

/// Loads the three sessions and validates everything that can be validated before the first phrase arrives.
fn load_model(
    encoder_path: &Path,
    decoder_path: &Path,
    joiner_path: &Path,
    tokens_path: &Path,
    debug: &DebugLog<'_>,
) -> Result<(OrtBackend, ModelInfo, bool), String> {
    // 1) tokens.txt first: it is cheap and a mismatch must stop everything before the 650 MB encoder is loaded.
    let tokens_text = std::fs::read_to_string(tokens_path).map_err(|e| format!("cannot read {}: {e}", tokens_path.display()))?;
    let vocab = Vocab::parse_strict(&tokens_text).map_err(|e| format!("tokens.txt rejected: {e}"))?;
    debug("info", &format!("tokens.txt OK: {} tokens, blank_id={} ({:?})", vocab.len(), vocab.blank_id, vocab.tokens[vocab.blank_id]));

    // 2) ONNX Runtime shared library (the ort crate is built with load-dynamic).
    crate::ort_runtime::ensure_loaded().map_err(|e| format!("ONNX Runtime library: {e}"))?;
    let ort_info = ort::info();
    let legacy_ort = pk::is_legacy_ort(ort_info);
    debug("info", &format!("ONNX Runtime library loaded (ensure_loaded OK); build info: {ort_info}"));
    if legacy_ort {
        debug(
            "warn",
            "this is an official ONNX Runtime 1.24.x build. Measured with the real int8 Parakeet model: ORT <= 1.24 returns an EMPTY text for \
             short speech fragments that ORT >= 1.25 transcribes, and is more sensitive to the Fbank recipe. The `test_onnx` front end is used \
             to compensate; the real fix is ORT 1.25.0 in scripts/fetch-onnxruntime.mjs (see the README of the patch).",
        );
    }

    // 3) sessions
    let encoder = load_ort_session(encoder_path).map_err(|e| format!("Encoder load failed: {e}"))?;
    debug("info", &describe_session("Encoder", &encoder));
    let decoder = load_ort_session(decoder_path).map_err(|e| format!("Decoder load failed: {e}"))?;
    debug("info", &describe_session("Decoder", &decoder));
    let joiner = load_ort_session(joiner_path).map_err(|e| format!("Joiner load failed: {e}"))?;
    debug("info", &describe_session("Joiner", &joiner));

    // 4) encoder contract: audio_signal f32 [B,128,T], length i64 [B]; outputs [B,1024,T'] f32 and lengths.
    if encoder.inputs().len() != 2 || encoder.outputs().len() < 2 {
        return Err(format!("Encoder must have 2 inputs and >= 2 outputs, found {} / {}", encoder.inputs().len(), encoder.outputs().len()));
    }
    let s = check_input(&encoder, 0, "Encoder", Dt::F32, 3)?;
    expect_dim("Encoder", "audio_signal", &s, 1, pk::FEAT_DIM)?;
    check_input(&encoder, 1, "Encoder", Dt::I64, 1)?;
    let s = check_output(&encoder, 0, "Encoder", Dt::F32, Some(3))?;
    expect_dim("Encoder", "output", &s, 1, pk::ENC_DIM)?;

    // 5) metadata
    let meta = encoder.metadata().map_err(|e| format!("NeMo metadata: {e}"))?;
    let normalize_type = meta.custom("normalize_type").unwrap_or_default();
    let meta_layers: Option<usize> = meta.custom("pred_rnn_layers").and_then(|s| s.parse().ok());
    let meta_hidden: Option<usize> = meta.custom("pred_hidden").and_then(|s| s.parse().ok());
    drop(meta);
    if !normalize_type.is_empty() && normalize_type != "per_feature" {
        return Err(format!("Unsupported NeMo normalize_type={normalize_type} (only per_feature is implemented)"));
    }

    // 6) decoder contract: targets i32 [B,U], target_length i32 [B], two f32 states [L,B,H]; outputs: f32 [B,H,U], lengths, 2 states.
    if decoder.inputs().len() != 4 || decoder.outputs().len() != 4 {
        return Err(format!("Decoder must have 4 inputs and 4 outputs, found {} / {}", decoder.inputs().len(), decoder.outputs().len()));
    }
    check_input(&decoder, 0, "Decoder", Dt::I32, 2)?;
    check_input(&decoder, 1, "Decoder", Dt::I32, 1)?;
    let st_a = check_input(&decoder, 2, "Decoder", Dt::F32, 3)?;
    let st_b = check_input(&decoder, 3, "Decoder", Dt::F32, 3)?;
    let layers_from_shape = st_a[0];
    let hidden_from_shape = st_a[2];
    if st_b[0] != layers_from_shape || st_b[2] != hidden_from_shape {
        return Err(format!("Decoder state inputs disagree: {st_a:?} vs {st_b:?}"));
    }
    let pick = |name: &str, meta: Option<usize>, shape: i64| -> Result<usize, String> {
        match (meta, shape) {
            (Some(m), s) if s > 0 && s as usize != m => Err(format!("{name}: encoder metadata says {m} but the decoder input shape says {s}")),
            (Some(m), _) => Ok(m),
            (None, s) if s > 0 => Ok(s as usize),
            _ => Err(format!("{name}: neither the metadata nor the decoder input shape defines it")),
        }
    };
    let pred_layers = pick("pred_rnn_layers", meta_layers, layers_from_shape)?;
    let pred_hidden = pick("pred_hidden", meta_hidden, hidden_from_shape)?;
    let s = check_output(&decoder, 0, "Decoder", Dt::F32, Some(3))?;
    expect_dim("Decoder", "output", &s, 1, pred_hidden)?;
    check_output(&decoder, 2, "Decoder", Dt::F32, Some(3))?;
    check_output(&decoder, 3, "Decoder", Dt::F32, Some(3))?;

    // 7) joiner contract and the vocabulary: logits = tokens + 5 TDT durations.
    if joiner.inputs().len() != 2 || joiner.outputs().len() != 1 {
        return Err(format!("Joiner must have 2 inputs and 1 output, found {} / {}", joiner.inputs().len(), joiner.outputs().len()));
    }
    let s = check_input(&joiner, 0, "Joiner", Dt::F32, 3)?;
    expect_dim("Joiner", "encoder_outputs", &s, 1, pk::ENC_DIM)?;
    let s = check_input(&joiner, 1, "Joiner", Dt::F32, 3)?;
    expect_dim("Joiner", "decoder_outputs", &s, 1, pred_hidden)?;
    let s = check_output(&joiner, 0, "Joiner", Dt::F32, None)?;
    if let Some(&last) = s.last() {
        if last > 0 && last as usize != vocab.len() + pk::N_DURATIONS {
            return Err(format!(
                "Joiner outputs {last} logits but tokens.txt has {} tokens: expected {} = {} tokens + {} TDT durations. \
                 tokens.txt does not belong to this model.",
                vocab.len(), vocab.len() + pk::N_DURATIONS, vocab.len(), pk::N_DURATIONS
            ));
        }
    }

    debug(
        "info",
        &format!(
            "model contract OK: encoder [1,{},T]->[1,{},T'], decoder states [{pred_layers},1,{pred_hidden}], joiner logits = {} tokens + {} durations, normalize_type={:?}",
            pk::FEAT_DIM, pk::ENC_DIM, vocab.len(), pk::N_DURATIONS, normalize_type
        ),
    );
    let info = ModelInfo { vocab, pred_layers, pred_hidden, per_feature_norm: normalize_type == "per_feature" };
    Ok((OrtBackend { encoder, decoder, joiner, pred_layers, pred_hidden }, info, legacy_ort))
}

// ─────────────────────────────────────────────────────────── glue ────────────────────────────────

fn variant_from_env(debug: &DebugLog<'_>, legacy_ort: bool) -> Variant {
    // An empty value (e.g. an unset workflow input) means "not set".
    let env_variant = std::env::var("NEXQ_PARAKEET_VARIANT").ok().filter(|s| !s.trim().is_empty());
    let env_tdt = std::env::var("NEXQ_PARAKEET_TDT").ok().filter(|s| !s.trim().is_empty());
    let (variant, log) = pk::select_variant(env_variant.as_deref(), env_tdt.as_deref(), legacy_ort);
    for (level, msg) in log {
        debug(level, &msg);
    }
    variant
}

pub(super) fn run(
    encoder_path: PathBuf,
    decoder_path: PathBuf,
    joiner_path: PathBuf,
    tokens_path: PathBuf,
    language: String,
    stop_flag: Arc<AtomicBool>,
    audio_rx: std::sync::mpsc::Receiver<AudioMessage>,
    result_tx: mpsc::Sender<TranscriptResult>,
    segment_counter_start: u64,
    debug: &DebugLog<'_>,
) {
    // A panic inside the thread would otherwise end it without a trace and the transcript would just stay empty.
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        run_inner(encoder_path, decoder_path, joiner_path, tokens_path, language, stop_flag, audio_rx, result_tx, segment_counter_start, debug)
    }));
    if let Err(p) = outcome {
        debug("error", &format!("NeMo thread panicked: {}", pk::panic_text(&p)));
    }
    debug("info", "NeMo inference thread exited");
}

fn run_inner(
    encoder_path: PathBuf,
    decoder_path: PathBuf,
    joiner_path: PathBuf,
    tokens_path: PathBuf,
    language: String,
    stop_flag: Arc<AtomicBool>,
    audio_rx: std::sync::mpsc::Receiver<AudioMessage>,
    result_tx: mpsc::Sender<TranscriptResult>,
    segment_counter_start: u64,
    debug: &DebugLog<'_>,
) {
    debug(
        "info",
        &format!(
            "NeMo Parakeet route selected (revision 3). files: encoder={} decoder={} joiner={} tokens={}",
            encoder_path.file_name().unwrap_or_default().to_string_lossy(),
            decoder_path.file_name().unwrap_or_default().to_string_lossy(),
            joiner_path.file_name().unwrap_or_default().to_string_lossy(),
            tokens_path.file_name().unwrap_or_default().to_string_lossy(),
        ),
    );
    let (mut backend, info, legacy_ort) = match load_model(&encoder_path, &decoder_path, &joiner_path, &tokens_path, debug) {
        Ok(m) => m,
        Err(err) => {
            debug("error", &format!("NeMo model loading failed — NO transcription will happen: {err}"));
            return;
        }
    };
    let variant = variant_from_env(debug, legacy_ort);

    let mut opts = pk::DriverOptions::new(variant, segment_counter_start);
    opts.dump_dir = std::env::var("NEXQ_PARAKEET_DUMP_DIR").ok().filter(|s| !s.is_empty()).map(PathBuf::from);

    let mut recv = |timeout: Duration| match audio_rx.recv_timeout(timeout) {
        Ok(AudioMessage::Samples(samples)) => pk::RecvEvent::Samples(samples),
        Ok(AudioMessage::Stop) => pk::RecvEvent::Stop,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => pk::RecvEvent::Timeout,
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => pk::RecvEvent::Disconnected,
    };
    let stopped = || stop_flag.load(Ordering::SeqCst);
    let mut emit = |e: pk::Emission| -> Result<(), String> {
        let result = TranscriptResult {
            text: e.text,
            is_final: true,
            confidence: 0.80,
            timestamp_ms: e.timestamp_ms,
            speaker: None,
            language: Some(language.clone()),
            segment_id: Some(e.segment_id),
            duration_ms: Some(e.duration_ms),
            speaker_tag: None,
        };
        result_tx.blocking_send(result).map_err(|err| format!("receiver closed: {err}"))
    };

    match pk::drive(&mut backend, &info, opts, &mut recv, &stopped, &mut emit, debug) {
        Ok(_report) => {}
        Err(err) => debug("error", &format!("NeMo Parakeet cannot start — NO transcription will happen: {err}")),
    }
}
