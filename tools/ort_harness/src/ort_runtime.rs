//! Same contract as NexQ's `ort_runtime::ensure_loaded()` (load-dynamic ONNX Runtime), with the library path taken
//! from `NEXQ_ORT_LIB` or, by default, from the copy staged by `scripts/fetch-onnxruntime.mjs`.
use std::path::PathBuf;
use std::sync::OnceLock;

fn library_path() -> PathBuf {
    if let Ok(p) = std::env::var("NEXQ_ORT_LIB") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    let name = if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("src-tauri").join("ort").join(name)
}

pub fn ensure_loaded() -> Result<(), String> {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        let library = library_path();
        if !library.exists() {
            return Err(format!("ONNX Runtime library not found: {} (set NEXQ_ORT_LIB)", library.display()));
        }
        ort::init_from(&library)
            .map_err(|e| format!("Failed to load ONNX Runtime from {}: {}", library.display(), e))?
            .commit();
        println!("[harness] ONNX Runtime loaded from {}", library.display());
        Ok(())
    })
    .clone()
}
