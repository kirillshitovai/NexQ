//! Stand-in for the part of NexQ's `stt/ort_streaming.rs` that the Parakeet modules depend on, plus the test driver.
//!
//! `AudioMessage` and `load_ort_session` are copies of the application's private items (same shape, same body), and the
//! two Parakeet modules are the *real files of the patch*, attached with the same `#[path]` lines the integration
//! patch adds to `ort_streaming.rs`. `run_production_route` calls `nemo_parakeet::run` exactly like
//! `inference_thread_main` does after the patch.
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::stt::provider::TranscriptResult;

#[path = "../../../../src-tauri/src/stt/nemo_parakeet_core.rs"]
mod nemo_parakeet_core;
#[path = "../../../../src-tauri/src/stt/nemo_parakeet.rs"]
mod nemo_parakeet;

pub use nemo_parakeet_core::{read_wav_mono16k, text_cer};

/// Same shape as the application's private enum.
enum AudioMessage {
    Samples(Vec<i16>),
    Stop,
}

/// Load an ORT session from a file (copy of the application's function, minus the `log` call).
fn load_ort_session(path: &std::path::Path) -> Result<ort::session::Session, String> {
    let filename = path.file_name().unwrap_or_default().to_string_lossy();

    crate::ort_runtime::ensure_loaded()?;
    let session = ort::session::Session::builder()
        .map_err(|e| format!("Session builder error: {}", e))?
        .with_intra_threads(4)
        .map_err(|e| format!("Thread config error: {}", e))?
        .commit_from_file(path)
        .map_err(|e| format!("Failed to load {}: {}", filename, e))?;

    println!("[harness] loaded {} ({} inputs, {} outputs)", filename, session.inputs().len(), session.outputs().len());
    Ok(session)
}

pub struct StreamRun {
    pub results: Vec<TranscriptResult>,
    pub logs: Vec<String>,
}

/// Feeds `audio` (16 kHz mono i16, `chunk` samples per message) to the Parakeet thread the way the application does:
/// chunks through the std channel, then the stop flag, then `Stop`; results come back through a tokio channel.
pub fn run_production_route(model_dir: &Path, tokens_override: Option<&Path>, audio: &[i16], chunk: usize) -> Result<StreamRun, String> {
    let encoder = model_dir.join("encoder.int8.onnx");
    let decoder = model_dir.join("decoder.int8.onnx");
    let joiner = model_dir.join("joiner.int8.onnx");
    let tokens = tokens_override.map(|p| p.to_path_buf()).unwrap_or_else(|| model_dir.join("tokens.txt"));

    let (audio_tx, audio_rx) = std::sync::mpsc::channel::<AudioMessage>();
    let (result_tx, mut result_rx) = tokio::sync::mpsc::channel::<TranscriptResult>(1024);
    let stop_flag = Arc::new(AtomicBool::new(false));
    let logs = Arc::new(Mutex::new(Vec::<String>::new()));

    let thread = std::thread::Builder::new()
        .name("ort-stt-inference".to_string())
        .spawn({
            let stop_flag = Arc::clone(&stop_flag);
            let logs = Arc::clone(&logs);
            move || {
                let debug = move |level: &str, msg: &str| {
                    println!("  [{level}] {msg}");
                    logs.lock().unwrap().push(format!("[{level}] {msg}"));
                };
                nemo_parakeet::run(encoder, decoder, joiner, tokens, "ru".to_string(), stop_flag, audio_rx, result_tx, 1, &debug);
            }
        })
        .map_err(|e| format!("cannot spawn the thread: {e}"))?;

    for c in audio.chunks(chunk) {
        if audio_tx.send(AudioMessage::Samples(c.to_vec())).is_err() {
            break; // the thread has already ended (model failed to load): the logs say why
        }
    }
    stop_flag.store(true, Ordering::SeqCst);
    let _ = audio_tx.send(AudioMessage::Stop);
    thread.join().map_err(|_| "the Parakeet thread panicked outside of its own catch_unwind".to_string())?;

    let mut results = Vec::new();
    while let Some(r) = result_rx.blocking_recv() {
        results.push(r);
    }
    let logs = logs.lock().unwrap().clone();
    Ok(StreamRun { results, logs })
}
