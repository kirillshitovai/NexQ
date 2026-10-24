//! Offline NeMo Parakeet TDT 0.6B v3 inference for NexQ 2.26.0.
//!
//! This path deliberately does not reuse Zipformer's streaming encoder/decoder
//! state managers: the NeMo FastConformer exports are offline transducers.
//! Performs one full inference per bounded speech segment.
use super::{load_ort_session, AudioMessage, SAMPLE_RATE};
use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use ort::value::Tensor;
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use crate::stt::provider::TranscriptResult;

const FEAT_DIM: usize = 128;
const FFT_SIZE: usize = 512;
const WINDOW_SAMPLES: usize = 400;
const FRAME_SHIFT: usize = 160;
const TAIL_PADDING: usize = 2 * SAMPLE_RATE as usize;
// These thresholds are intentionally conservative; they govern segmentation,
// not ONNX decoding. Adjust them if device recordings are unusually quiet.
const START_RMS: f32 = 0.004;
const CONTINUE_RMS: f32 = 0.0025;
const SILENCE_SAMPLES: usize = (SAMPLE_RATE as usize) * 8 / 10;
const MAX_SEGMENT_SAMPLES: usize = (SAMPLE_RATE as usize) * 10;
const MIN_SEGMENT_SAMPLES: usize = (SAMPLE_RATE as usize) * 3 / 10;
const PREROLL_SAMPLES: usize = (SAMPLE_RATE as usize) / 5;

type DebugLog<'a> = dyn Fn(&str, &str) + 'a;

struct Model {
    encoder: Session,
    decoder: Session,
    joiner: Session,
    tokens: HashMap<i64, String>,
    blank_id: i64,
    normalize_per_feature: bool,
    pred_layers: usize,
    pred_hidden: usize,
}

impl Model {
    fn load(
        encoder_path: &PathBuf,
        decoder_path: &PathBuf,
        joiner_path: &PathBuf,
        tokens_path: &PathBuf,
    ) -> Result<Self, String> {
        let encoder = load_ort_session(encoder_path)?;
        let decoder = load_ort_session(decoder_path)?;
        let joiner = load_ort_session(joiner_path)?;
        let tokens = super::load_tokens(tokens_path)?;
        if tokens.is_empty() {
            return Err("NeMo tokens.txt is empty".into());
        }
        // NeMo's blank is the *last* vocabulary entry, unlike Zipformer.
        let blank_id = (tokens.len() - 1) as i64;
        if !tokens.contains_key(&blank_id) {
            return Err(format!("NeMo blank_id {} is missing in tokens.txt", blank_id));
        }
        let meta = encoder.metadata().map_err(|e| format!("NeMo metadata: {e}"))?;
        let normalize_type = meta.custom("normalize_type").unwrap_or_default();
        if !normalize_type.is_empty() && normalize_type != "per_feature" {
            return Err(format!("Unexpected NeMo normalize_type={normalize_type}"));
        }
        let pred_layers = meta.custom("pred_rnn_layers")
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);
        let pred_hidden = meta.custom("pred_hidden")
            .and_then(|s| s.parse().ok())
            .unwrap_or(640);
        drop(meta);
        if encoder.inputs().len() != 2 || encoder.outputs().len() < 2 {
            return Err("Parakeet encoder must expose audio_signal, length and 2 outputs".into());
        }
        if encoder.inputs()[0].name() != "audio_signal" || encoder.inputs()[1].name() != "length" {
            return Err(format!("Unexpected NeMo Encoder inputs: {:?}, {:?}",
                encoder.inputs()[0].name(), encoder.inputs()[1].name()));
        }
        if decoder.inputs().len() != 4 || decoder.outputs().len() != 4 {
            return Err("Parakeet decoder must have 4 inputs and 4 outputs".into());
        }
        if joiner.inputs().len() != 2 || joiner.outputs().len() != 1 {
            return Err("Parakeet joiner must have 2 inputs and 1 output".into());
        }
        Ok(Self {
            encoder, decoder, joiner, tokens, blank_id,
            normalize_per_feature: normalize_type == "per_feature",
            pred_layers, pred_hidden,
        })
    }

    fn decode_segment(&mut self, audio: &[f32]) -> Result<String, String> {
        let mut padded = Vec::with_capacity(audio.len() + TAIL_PADDING);
        padded.extend_from_slice(audio);
        // Tail padding is added once at finalization, before FBANK extraction.
        padded.resize(padded.len() + TAIL_PADDING, 0.0);
        let mut features = fbank_128_librosa(&padded)?;
        if self.normalize_per_feature {
            normalize_features(&mut features);
        }
        let time = features.len();
        if time == 0 {
            return Ok(String::new());
        }
        let mut nch = vec![0.0f32; FEAT_DIM * time];
        for t in 0..time {
            for c in 0..FEAT_DIM {
                nch[c * time + t] = features[t][c];
            }
        }
        let signal = Tensor::from_array(
            Array3::from_shape_vec((1, FEAT_DIM, time), nch)
                .map_err(|e| format!("NeMo feature shape: {e}"))?
        ).map_err(|e| format!("NeMo audio tensor: {e}"))?;
        let length = Tensor::from_array(Array1::from_vec(vec![time as i64]))
            .map_err(|e| format!("NeMo length tensor: {e}"))?;
        let outputs = self.encoder.run(ort::inputs![signal, length])
            .map_err(|e| format!("NeMo encoder: {e}"))?;
        let (shape, enc_data) = outputs[0].try_extract_tensor::<f32>()
            .map_err(|e| format!("NeMo encoder output: {e}"))?;
        if shape.len() != 3 || shape[0] != 1 || shape[1] != 1024 {
            return Err(format!("NeMo encoder expected [1,1024,T], got {shape:?}"));
        }
        let out_frames = shape[2] as usize;
        if enc_data.len() != 1024 * out_frames {
            return Err(format!("NeMo encoder data length mismatch: {}", enc_data.len()));
        }
        // Own the Encoder output before mutably borrowing the Decoder session.
        let enc_data = enc_data.to_vec();
        drop(outputs);
        let mut state0 = vec![0.0f32; self.pred_layers * self.pred_hidden];
        let mut state1 = vec![0.0f32; self.pred_layers * self.pred_hidden];
        let (mut decoder_out, mut next0, mut next1) =
            self.decoder_step(self.blank_id, &state0, &state1)?;
        let vocab_size = self.tokens.len();
        let mut decoded = Vec::<i64>::new();
        let mut t = 0;
        while t < out_frames {
            let mut enc_frame = vec![0.0f32; 1024];
            for c in 0..1024 {
                enc_frame[c] = enc_data[c * out_frames + t];
            }
            let enc = Tensor::from_array(Array3::from_shape_vec((1, 1024, 1), enc_frame)
                .map_err(|e| format!("NeMo joiner encoder shape: {e}"))?)
                .map_err(|e| format!("NeMo joiner encoder tensor: {e}"))?;
            // NeMo joiner inputs are [1,dim,1], NOT [1,1,dim].
            let dec = Tensor::from_array(Array3::from_shape_vec(
                (1, self.pred_hidden, 1), decoder_out.clone()
            ).map_err(|e| format!("NeMo joiner decoder shape: {e}"))?)
                .map_err(|e| format!("NeMo joiner decoder tensor: {e}"))?;
            let jo = self.joiner.run(ort::inputs![enc, dec])
                .map_err(|e| format!("NeMo joiner: {e}"))?;
            let (_, logits) = jo[0].try_extract_tensor::<f32>()
                .map_err(|e| format!("NeMo logits: {e}"))?;
            if logits.len() <= vocab_size {
                return Err(format!("NeMo logits expected token+duration (> {vocab_size}), got {}", logits.len()));
            }
            let token = argmax(&logits[..vocab_size]) as i64;
            let skip = argmax(&logits[vocab_size..]).max(1);
            // SessionOutputs borrows the Joiner; release it before updating Decoder.
            drop(jo);
            if token != self.blank_id {
                decoded.push(token);
                // Commit states only when a token was emitted, as in sherpa-onnx.
                state0 = next0;
                state1 = next1;
                (decoder_out, next0, next1) = self.decoder_step(token, &state0, &state1)?;
            }
            t += skip;
        }
        let mut text = String::new();
        for token in decoded {
            if let Some(part) = self.tokens.get(&token) {
                text.push_str(part);
            }
        }
        Ok(text.replace('▁', " ").trim().to_string())
    }

    fn decoder_step(
        &mut self, token: i64, state0: &[f32], state1: &[f32]
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let target = Tensor::from_array(Array2::from_shape_vec((1, 1), vec![token as i32])
            .map_err(|e| format!("NeMo target shape: {e}"))?)
            .map_err(|e| format!("NeMo target tensor: {e}"))?;
        let target_len = Tensor::from_array(Array1::from_vec(vec![1i32]))
            .map_err(|e| format!("NeMo target length: {e}"))?;
        let s0 = Tensor::from_array(Array3::from_shape_vec(
            (self.pred_layers, 1, self.pred_hidden), state0.to_vec()
        ).map_err(|e| format!("NeMo state0 shape: {e}"))?)
            .map_err(|e| format!("NeMo state0 tensor: {e}"))?;
        let s1 = Tensor::from_array(Array3::from_shape_vec(
            (self.pred_layers, 1, self.pred_hidden), state1.to_vec()
        ).map_err(|e| format!("NeMo state1 shape: {e}"))?)
            .map_err(|e| format!("NeMo state1 tensor: {e}"))?;
        let outs = self.decoder.run(ort::inputs![target, target_len, s0, s1])
            .map_err(|e| format!("NeMo decoder: {e}"))?;
        let (_, dec) = outs[0].try_extract_tensor::<f32>()
            .map_err(|e| format!("NeMo decoder output: {e}"))?;
        let (_, n0) = outs[2].try_extract_tensor::<f32>()
            .map_err(|e| format!("NeMo decoder state0: {e}"))?;
        let (_, n1) = outs[3].try_extract_tensor::<f32>()
            .map_err(|e| format!("NeMo decoder state1: {e}"))?;
        let expected_states = self.pred_layers * self.pred_hidden;
        if dec.len() != self.pred_hidden || n0.len() != expected_states || n1.len() != expected_states {
            return Err(format!(
                "NeMo decoder dimensions wrong: embed={} (want {}), states={}/{} (want {})",
                dec.len(), self.pred_hidden, n0.len(), n1.len(), expected_states
            ));
        }
        Ok((dec.to_vec(), n0.to_vec(), n1.to_vec()))
    }
}

fn argmax(xs: &[f32]) -> usize {
    xs.iter().enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(idx, _)| idx)
}

fn normalize_features(features: &mut [Vec<f32>]) {
    let n = features.len();
    if n <= 1 { return; }
    for f in 0..FEAT_DIM {
        let mean = features.iter().map(|row| row[f]).sum::<f32>() / n as f32;
        let variance = features.iter().map(|row| (row[f] - mean).powi(2)).sum::<f32>() / (n - 1) as f32;
        let denom = variance.sqrt() + 1e-5;
        for row in features.iter_mut() {
            row[f] = (row[f] - mean) / denom;
        }
    }
}

// Kaldi-native-fbank compatible parameters for NeMo (16k, 25ms/10ms,
// 128 mel bins, Hann, no dither/DC removal, is_librosa=true). The librosa
// Slaney mel scale and area-normalized triangles differ from NexQ's HTK FBANK.
fn fbank_128_librosa(audio: &[f32]) -> Result<Vec<Vec<f32>>, String> {
    if audio.len() < WINDOW_SAMPLES { return Ok(Vec::new()); }
    let mut filters = vec![vec![0.0f32; FFT_SIZE / 2 + 1]; FEAT_DIM];
    let mel_max = librosa_hz_to_mel(SAMPLE_RATE as f32 / 2.0);
    let hz_edges: Vec<f32> = (0..FEAT_DIM + 2)
        .map(|i| librosa_mel_to_hz(mel_max * i as f32 / (FEAT_DIM + 1) as f32))
        .collect();
    for m in 0..FEAT_DIM {
        let (left, center, right) = (hz_edges[m], hz_edges[m + 1], hz_edges[m + 2]);
        let norm = 2.0 / (right - left);
        for k in 0..=FFT_SIZE / 2 {
            let hz = (k as f32) * SAMPLE_RATE as f32 / FFT_SIZE as f32;
            let weight = if hz >= left && hz <= center {
                (hz - left) / (center - left)
            } else if hz > center && hz <= right {
                (right - hz) / (right - center)
            } else { 0.0 };
            filters[m][k] = weight.max(0.0) * norm;
        }
    }
    let hann: Vec<f32> = (0..WINDOW_SAMPLES).map(|i|
        0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (WINDOW_SAMPLES - 1) as f32).cos()
    ).collect();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let nframes = 1 + (audio.len() - WINDOW_SAMPLES) / FRAME_SHIFT;
    let mut result = Vec::with_capacity(nframes);
    for i in 0..nframes {
        let window = &audio[i * FRAME_SHIFT..i * FRAME_SHIFT + WINDOW_SAMPLES];
        let mut tmp: Vec<f32> = window.to_vec();
        // Kaldi pre-emphasis with first sample duplicated (default 0.97).
        for n in (1..WINDOW_SAMPLES).rev() { tmp[n] -= 0.97 * tmp[n - 1]; }
        tmp[0] -= 0.97 * tmp[0];
        let mut spectrum = vec![Complex::new(0.0f32, 0.0f32); FFT_SIZE];
        for n in 0..WINDOW_SAMPLES { spectrum[n].re = tmp[n] * hann[n]; }
        fft.process(&mut spectrum);
        let power: Vec<f32> = spectrum[..=FFT_SIZE / 2].iter().map(|z| z.norm_sqr()).collect();
        let mel = filters.iter().map(|filter| {
            let energy = power.iter().zip(filter).map(|(p, w)| p * w).sum::<f32>();
            energy.max(f32::EPSILON).ln()
        }).collect();
        result.push(mel);
    }
    Ok(result)
}

fn librosa_hz_to_mel(freq: f32) -> f32 {
    let linear_hz_per_mel = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / linear_hz_per_mel;
    if freq < min_log_hz { freq / linear_hz_per_mel }
    else { min_log_mel + (freq / min_log_hz).ln() / (6.4_f32.ln() / 27.0) }
}

fn librosa_mel_to_hz(mel: f32) -> f32 {
    let min_log_mel = 15.0;
    if mel < min_log_mel { mel * (200.0 / 3.0) }
    else { 1000.0 * ((mel - min_log_mel) * (6.4_f32.ln() / 27.0)).exp() }
}

fn emit_segment(
    model: &mut Model, segment: &mut Vec<f32>, language: &str,
    tx: &mpsc::Sender<TranscriptResult>, segment_id: &mut u64, debug: &DebugLog<'_>,
) {
    if segment.len() < MIN_SEGMENT_SAMPLES { segment.clear(); return; }
    match model.decode_segment(segment) {
        Ok(text) => {
            if !text.is_empty() {
                let timestamp_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
                let result = TranscriptResult {
                    text,
                    is_final: true,
                    confidence: 0.80,
                    timestamp_ms,
                    speaker: None,
                    language: Some(language.to_string()),
                    segment_id: Some(format!("ort_{}", *segment_id)),
                    duration_ms: Some((segment.len() as u64) * 1000 / SAMPLE_RATE as u64),
                    speaker_tag: None,
                };
                if tx.blocking_send(result).is_err() { debug("warn", "NeMo transcript receiver closed"); }
                else { debug("info", "NeMo emitted final transcript"); }
            } else { debug("info", "NeMo decoded an empty segment"); }
        }
        Err(error) => debug("error", &format!("NeMo segment inference failed: {error}")),
    }
    *segment_id += 1;
    segment.clear();
}

pub(super) fn run(
    encoder_path: PathBuf, decoder_path: PathBuf, joiner_path: PathBuf,
    tokens_path: PathBuf, language: String, stop_flag: Arc<AtomicBool>,
    audio_rx: std::sync::mpsc::Receiver<AudioMessage>,
    result_tx: mpsc::Sender<TranscriptResult>,
    segment_counter_start: u64, debug: &DebugLog<'_>,
) {
    let mut model = match Model::load(&encoder_path, &decoder_path, &joiner_path, &tokens_path) {
        Ok(m) => m,
        Err(err) => { debug("error", &format!("NeMo model loading failed: {err}")); return; }
    };
    debug("info", &format!("NeMo Parakeet ready: vocab={}, blank={}, per_feature={}",
        model.tokens.len(), model.blank_id, model.normalize_per_feature));
    let mut segment = Vec::<f32>::new();
    let mut preroll = VecDeque::<f32>::with_capacity(PREROLL_SAMPLES);
    let mut pending = Vec::<f32>::new();
    let mut consecutive_silence = 0usize;
    let mut seg_id = segment_counter_start;
    loop {
        match audio_rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(AudioMessage::Samples(samples)) => {
                pending.extend(samples.into_iter().map(|x| x as f32 / 32768.0));
            }
            Ok(AudioMessage::Stop) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if stop_flag.load(Ordering::SeqCst) { break; }
                continue;
            }
        }
        // Segment on 10 ms windows so the endpoint is independent of the audio
        // sender's chunk size; keep 200 ms preroll to preserve initial consonants.
        let complete = pending.len() / FRAME_SHIFT * FRAME_SHIFT;
        for block in pending[..complete].chunks_exact(FRAME_SHIFT) {
            let rms = (block.iter().map(|x| x * x).sum::<f32>() / block.len() as f32).sqrt();
            if segment.is_empty() {
                if rms > START_RMS {
                    segment.extend(preroll.drain(..));
                    segment.extend_from_slice(block);
                    consecutive_silence = 0;
                } else {
                    for &x in block {
                        preroll.push_back(x);
                        if preroll.len() > PREROLL_SAMPLES { preroll.pop_front(); }
                    }
                }
            } else {
                segment.extend_from_slice(block);
                if rms > CONTINUE_RMS { consecutive_silence = 0; }
                else { consecutive_silence += block.len(); }
                if consecutive_silence >= SILENCE_SAMPLES || segment.len() >= MAX_SEGMENT_SAMPLES {
                    emit_segment(&mut model, &mut segment, &language, &result_tx, &mut seg_id, debug);
                    consecutive_silence = 0;
                    preroll.clear();
                }
            }
        }
        pending.drain(..complete);
    }
    if !segment.is_empty() {
        segment.extend_from_slice(&pending);
        emit_segment(&mut model, &mut segment, &language, &result_tx, &mut seg_id, debug);
    }
    debug("info", "NeMo inference thread exited");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn librosa_frequency_round_trip() {
        for hz in [0.0f32, 200.0, 1000.0, 2500.0, 8000.0] {
            let round = librosa_mel_to_hz(librosa_hz_to_mel(hz));
            assert!((round - hz).abs() < hz.max(1.0) * 0.00001);
        }
    }
    #[test]
    fn fbank_has_expected_shape_and_finite_values() {
        let audio = vec![0.0f32; 16000];
        let feat = fbank_128_librosa(&audio).unwrap();
        assert_eq!(feat.len(), 1 + (16000 - WINDOW_SAMPLES) / FRAME_SHIFT);
        assert!(feat.iter().all(|f| f.len() == FEAT_DIM && f.iter().all(|v| v.is_finite())));
    }
    #[test]
    fn normalization_is_per_feature() {
        let mut feat = vec![vec![1.0f32; FEAT_DIM], vec![3.0f32; FEAT_DIM], vec![5.0f32; FEAT_DIM]];
        normalize_features(&mut feat);
        assert!(feat.iter().all(|row| row.iter().all(|value| value.is_finite())));
        assert!(feat[1][3].abs() < 1e-5);
    }
    // Run explicitly on Windows: cargo test real_russian_wav -- --ignored --nocapture
    #[test]
    #[ignore = "requires downloaded Parakeet ONNX files, ONNX runtime and WAV"]
    fn real_russian_wav() {
        let dir = std::path::PathBuf::from(std::env::var("APPDATA").expect("APPDATA"))
            .join("com.nexq.app/models/parakeet_tdt/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8");
        let file = std::env::var("NEXQ_TEST_WAV").expect("Set NEXQ_TEST_WAV to a mono, PCM16, 16kHz Russian WAV");
        let wav = hound::WavReader::open(&file).expect("WAV open");
        let spec = wav.spec();
        assert_eq!(spec.sample_rate, 16000);
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.bits_per_sample, 16);
        assert_eq!(spec.sample_format, hound::SampleFormat::Int);
        let audio: Vec<f32> = wav.into_samples::<i16>()
            .map(|s| s.expect("PCM sample") as f32 / 32768.0).collect();
        let mut model = Model::load(
            &dir.join("encoder.int8.onnx"),
            &dir.join("decoder.int8.onnx"),
            &dir.join("joiner.int8.onnx"),
            &dir.join("tokens.txt"),
        ).expect("Parakeet ONNX model load");
        let text = model.decode_segment(&audio).expect("Parakeet TDT inference");
        println!("Transcription: {text}");
        assert!(!text.is_empty(), "Parakeet returned empty transcription");
        assert!(text.chars().any(|c| ('\u{0400}'..='\u{04FF}').contains(&c)),
            "No Cyrillic in recognized text");
        if let Ok(expected) = std::env::var("NEXQ_TEST_EXPECTED") {
            assert!(text.to_lowercase().contains(&expected.to_lowercase()),
                "Expected phrase {expected:?} not found in {text:?}");
        }
    }

}
