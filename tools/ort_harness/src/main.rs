//! Real-model check of the NeMo Parakeet patch (no Tauri). See README.md.
//!
//! Environment:
//!   NEXQ_MODEL_DIR     folder with encoder.int8.onnx, decoder.int8.onnx, joiner.int8.onnx, tokens.txt
//!   NEXQ_WAV_DIR       folder with 16 kHz PCM16 WAV files
//!   NEXQ_EXPECT_FILE   lines "wav-file-stem|expected text"; only WAVs listed there are used
//!   NEXQ_ORT_LIB       onnxruntime.dll / libonnxruntime.so (default: src-tauri/ort/…)
//!   NEXQ_MAX_CER       pass threshold for the whole stream (default 0.10)
//!   NEXQ_CHUNKS        samples per audio message, comma separated (default "441,4096")
//!   NEXQ_PARAKEET_VARIANT / NEXQ_PARAKEET_TDT   same switches as in the application
//! Exit code 0 only when every check passed.
mod ort_runtime;
mod stt;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use stt::ort_streaming::{read_wav_mono16k, run_production_route, text_cer};

fn env_path(name: &str) -> Result<PathBuf, String> {
    std::env::var(name).map(PathBuf::from).map_err(|_| format!("environment variable {name} is not set"))
}

/// Faint broadband noise (rms ~0.0008, like a quiet room) between the phrases.
fn quiet(secs: f32, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..(16000.0 * secs) as usize)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0;
            u * 0.0008 * 3.0f32.sqrt()
        })
        .collect()
}

fn to_i16(x: &[f32]) -> Vec<i16> {
    x.iter().map(|&v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16).collect()
}

fn expectations(path: &Path) -> Result<BTreeMap<String, String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(text
        .lines()
        .filter_map(|l| l.split_once('|'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect())
}

fn main() {
    match real_main() {
        Ok(true) => {
            println!("\nALL CHECKS PASSED");
        }
        Ok(false) => {
            println!("\nCHECKS FAILED");
            std::process::exit(1);
        }
        Err(e) => {
            println!("\nERROR: {e}");
            std::process::exit(2);
        }
    }
}

fn real_main() -> Result<bool, String> {
    let model_dir = env_path("NEXQ_MODEL_DIR")?;
    let wav_dir = env_path("NEXQ_WAV_DIR")?;
    let expected = expectations(&env_path("NEXQ_EXPECT_FILE")?)?;
    let max_cer: f32 = std::env::var("NEXQ_MAX_CER").ok().and_then(|v| v.parse().ok()).unwrap_or(0.10);
    let chunks: Vec<usize> = std::env::var("NEXQ_CHUNKS")
        .unwrap_or_else(|_| "441,4096".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .filter(|&c| c > 0)
        .collect();
    if chunks.is_empty() {
        return Err("NEXQ_CHUNKS has no valid chunk size".into());
    }

    let mut wavs: Vec<PathBuf> = std::fs::read_dir(&wav_dir)
        .map_err(|e| format!("{}: {e}", wav_dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |x| x.eq_ignore_ascii_case("wav")))
        .filter(|p| expected.contains_key(&p.file_stem().unwrap_or_default().to_string_lossy().to_string()))
        .collect();
    wavs.sort();
    if wavs.is_empty() {
        return Err("no WAV in NEXQ_WAV_DIR has an expected text in NEXQ_EXPECT_FILE".into());
    }

    let mut audio = quiet(1.0, 100);
    let mut reference = Vec::new();
    for (i, w) in wavs.iter().enumerate() {
        audio.extend(read_wav_mono16k(w)?);
        reference.push(expected[&w.file_stem().unwrap_or_default().to_string_lossy().to_string()].clone());
        if i + 1 < wavs.len() {
            audio.extend(quiet(1.2, 200 + i as u64));
        }
    }
    let reference = reference.join(" ");
    let audio = to_i16(&audio);
    println!("stream: {} phrases, {:.1} s; reference: {reference}", wavs.len(), audio.len() as f32 / 16000.0);

    let mut ok = true;

    // 1) a wrong tokens.txt must be a hard, visible error before any model is loaded
    {
        println!("\n=== check 1: tokens.txt with a missing line is a hard error ===");
        let text = std::fs::read_to_string(model_dir.join("tokens.txt")).map_err(|e| format!("tokens.txt: {e}"))?;
        let mut lines: Vec<&str> = text.lines().collect();
        lines.pop();
        let bad = std::env::temp_dir().join(format!("nexq_bad_tokens_{}.txt", std::process::id()));
        std::fs::write(&bad, lines.join("\n") + "\n").map_err(|e| format!("{}: {e}", bad.display()))?;
        let run = run_production_route(&model_dir, Some(&bad), &audio[..16000], 441)?;
        let _ = std::fs::remove_file(&bad);
        let said_so = run.logs.iter().any(|l| l.starts_with("[error]") && l.contains("NO transcription"));
        let passed = run.results.is_empty() && said_so;
        println!("-> results={} error logged={} : {}", run.results.len(), said_so, if passed { "PASS" } else { "FAIL" });
        ok &= passed;
    }

    // 2) the real stream, once per sender chunk size
    let mut joined_all: Vec<String> = Vec::new();
    for &chunk in &chunks {
        println!("\n=== check 2: production route, {chunk} samples per message ===");
        let t0 = std::time::Instant::now();
        let run = run_production_route(&model_dir, None, &audio, chunk)?;
        for r in &run.results {
            println!(
                "  result {} final={} {:5.2}s | {}",
                r.segment_id.clone().unwrap_or_default(),
                r.is_final,
                r.duration_ms.unwrap_or(0) as f32 / 1000.0,
                r.text
            );
        }
        let joined = run.results.iter().map(|r| r.text.clone()).collect::<Vec<_>>().join(" ");
        let cer = text_cer(&joined, &reference);
        let errors: Vec<&String> = run.logs.iter().filter(|l| l.starts_with("[error]")).collect();
        let mut ids: Vec<String> = run.results.iter().filter_map(|r| r.segment_id.clone()).collect();
        ids.sort();
        ids.dedup();
        let checks: Vec<(String, bool)> = vec![
            ("at least one result was delivered".to_string(), !run.results.is_empty()),
            ("every result is final and has a segment id".to_string(), run.results.iter().all(|r| r.is_final && r.segment_id.is_some())),
            ("segment ids are unique".to_string(), ids.len() == run.results.len()),
            ("no [error] in the log".to_string(), errors.is_empty()),
            (format!("stream CER {:.1}% <= {:.1}%", cer * 100.0, max_cer * 100.0), cer <= max_cer),
        ];
        for (what, passed) in checks {
            println!("-> {} : {}", what, if passed { "PASS" } else { "FAIL" });
            ok &= passed;
        }
        println!("   ({} results, {:.1} s wall time)", run.results.len(), t0.elapsed().as_secs_f32());
        joined_all.push(joined);
    }
    let same = joined_all.windows(2).all(|w| w[0] == w[1]);
    println!("-> transcript does not depend on the sender's chunk size : {}", if same { "PASS" } else { "FAIL" });
    ok &= same;
    Ok(ok)
}
