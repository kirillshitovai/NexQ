#!/usr/bin/env node
// NexQ v2.26.0 — NeMo Parakeet TDT patch, revision 3: installer.
//
//   node parakeet-patch/apply-patch.mjs --project-root <NexQ repository root> [--skip-ort-bump] [--dry-run]
//
// Needed only when the patch ZIP was NOT simply unpacked over the repository root (for example when ort_streaming.rs
// has local changes that must not be overwritten). Run it from the unpacked ZIP; the module files are taken from
// <ZIP>/src-tauri/src/stt/.
//
// What it does (and nothing else):
//   1. copies src-tauri/src/stt/nemo_parakeet.rs and nemo_parakeet_core.rs into the project;
//   2. adds the two `#[path]` module lines and the Parakeet branch to src-tauri/src/stt/ort_streaming.rs
//      (a backup is kept as ort_streaming.rs.bak-before-parakeet-fix; trees patched by revision 1 or 2 are upgraded);
//   3. moves scripts/fetch-onnxruntime.mjs from ONNX Runtime 1.24.2 to 1.25.0 (skip with --skip-ort-bump).
//      Reason: measured in this patch's README — the INT8 Parakeet encoder returns empty text on short utterances
//      with ONNX Runtime <= 1.24.x and decodes them with >= 1.25.0.
// It is idempotent and refuses to touch a tree it does not recognise.
import { copyFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url)); // <ZIP>/parakeet-patch
const zipRoot = resolve(here, "..");
const argv = process.argv.slice(2);
const flag = (name) => argv.includes(name);
const option = (name, fallback) => {
  const i = argv.indexOf(name);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback;
};
const projectRoot = resolve(option("--project-root", "."));
const skipOrtBump = flag("--skip-ort-bump");
const dryRun = flag("--dry-run");

const fail = (msg) => {
  console.error(`ERROR: ${msg}\nNo file was changed.`);
  process.exit(1);
};
const warn = (msg) => console.warn(`WARNING: ${msg}`);
const info = (msg) => console.log(msg);

const cargoToml = join(projectRoot, "src-tauri", "Cargo.toml");
const sttDir = join(projectRoot, "src-tauri", "src", "stt");
const source = join(sttDir, "ort_streaming.rs");
const fetchScript = join(projectRoot, "scripts", "fetch-onnxruntime.mjs");
const modules = ["nemo_parakeet.rs", "nemo_parakeet_core.rs"];

if (!existsSync(cargoToml) || !existsSync(source)) fail("src-tauri/Cargo.toml or src-tauri/src/stt/ort_streaming.rs not found: set --project-root to the NexQ repository root.");
for (const m of modules) {
  if (!existsSync(join(zipRoot, "src-tauri", "src", "stt", m))) fail(`src-tauri/src/stt/${m} is missing in the unpacked ZIP (incomplete ZIP).`);
}
const cargo = readFileSync(cargoToml, "utf8");
if (!/^version\s*=\s*"2\.26\.0"\s*$/m.test(cargo)) fail('this patch targets NexQ v2.26.0 only (version in src-tauri/Cargo.toml differs).');
for (const dep of ["ndarray", "ort"]) {
  if (!new RegExp(`^\\s*${dep}(\\s*=|\\.)`, "m").test(cargo) && !new RegExp(`^\\[(dev-|build-)?dependencies\\.${dep}\\]`, "m").test(cargo)) {
    warn(`dependency '${dep}' was not found in Cargo.toml; cargo check will fail.`);
  }
}

// ── ort_streaming.rs ───────────────────────────────────────────────────────────────────────────────────────────────
const original = readFileSync(source, "utf8");
const nl = original.includes("\r\n") ? "\r\n" : "\n";
const lf = (s) => s.replace(/\r\n/g, "\n").replace(/\n/g, nl);
const count = (hay, needle) => hay.split(needle).length - 1;

const importAnchor = "use crate::stt::provider::{STTProvider, STTProviderType, TranscriptResult};";
const branchAnchor = '    debug("info", "Inference thread started, loading models...");';
const glueMod = '#[path = "nemo_parakeet.rs"]\nmod nemo_parakeet;';
const coreMod = '#[path = "nemo_parakeet_core.rs"]\nmod nemo_parakeet_core;';
// Both blocks are inserted after an existing line that is followed by a blank line, so the result is identical to
// ort_streaming.integration.patch.
const modsBlock = lf(
  "// NeMo Parakeet TDT (offline transducer): front end, TDT decoding, segmenter and stream driver live in\n" +
    "// these two modules; this file only routes Parakeet models to them (see nemo_parakeet.rs).\n" +
    coreMod + "\n" + glueMod
);
const branch = lf(
  "    // NeMo Parakeet TDT is an offline transducer: it needs its own front end, TDT decoding and\n" +
    "    // segmentation, not the streaming Zipformer pipeline below.\n" +
    '    if _model_dir.to_string_lossy().to_ascii_lowercase().contains("parakeet") {\n' +
    "        nemo_parakeet::run(\n" +
    "            encoder_path,\n            decoder_path,\n            joiner_path,\n            tokens_path,\n" +
    "            language,\n            stop_flag,\n            audio_rx,\n            result_tx,\n" +
    "            segment_counter_start,\n            &debug,\n        );\n        return;\n    }"
);

const hasGlue = original.includes("mod nemo_parakeet;");
const hasCore = original.includes("mod nemo_parakeet_core;");
const hasBranch = original.includes("nemo_parakeet::run(");
let patched = null;
let state;
if (!hasGlue && !hasCore && !hasBranch) {
  state = "clean v2.26.0";
  if (count(original, importAnchor) !== 1 || count(original, branchAnchor) !== 1) fail("the expected v2.26.0 anchors in ort_streaming.rs were not found exactly once.");
  patched = original
    .replace(importAnchor, () => importAnchor + nl + nl + modsBlock)
    .replace(branchAnchor, () => branchAnchor + nl + nl + branch);
} else if (hasGlue && hasBranch && !hasCore) {
  state = "patched by revision 1/2 (no core module yet)";
  const glueLine = lf(glueMod);
  if (count(original, glueLine) !== 1) fail("found 'mod nemo_parakeet;' but not in the expected '#[path]' form: restore ort_streaming.rs from git and run again.");
  patched = original.replace(glueLine, () => lf(coreMod) + nl + glueLine);
} else if (hasGlue && hasBranch && hasCore) {
  state = "already installed";
} else {
  fail("ort_streaming.rs is partially patched (module lines and the Parakeet branch do not match). Restore it with `git checkout -- src-tauri/src/stt/ort_streaming.rs` (or from the .bak-before-parakeet-fix file) and run again.");
}
info(`ort_streaming.rs: ${state}`);

// ── fetch-onnxruntime.mjs ──────────────────────────────────────────────────────────────────────────────────────────
let fetchPatched = null;
if (!skipOrtBump) {
  if (!existsSync(fetchScript)) {
    warn("scripts/fetch-onnxruntime.mjs not found; ONNX Runtime was not bumped.");
  } else {
    const text = readFileSync(fetchScript, "utf8");
    const fnl = text.includes("\r\n") ? "\r\n" : "\n";
    const OLD = {
      version: 'const ORT_VERSION = "1.24.2";',
      win: 'const WIN_ZIP_SHA256 = "8e3e9c826375352e29cb2614fe44f3d7a4b0ff7b8028ad7a456af9d949a7e8b0";',
      mac: 'const MAC_ARM64_TGZ_SHA256 = "0af4fa503e8ea285245b47ee42d0a7461b8156a81270857da0c1d4ecf858abde";',
      macComment: "// macOS (Microsoft publishes arm64 only for 1.24.2 — Apple Silicon)",
    };
    const NEW = {
      version:
        [
          "// 1.25.0 rather than 1.24.2: the INT8 NeMo Parakeet TDT encoder returns an empty transcript for",
          "// short utterances that start abruptly on ONNX Runtime <= 1.24.x, and decodes them on >= 1.25.0",
          "// (measured with the same features, model and decoder loop; see docs of the Parakeet patch).",
          'const ORT_VERSION = "1.25.0";',
        ].join(fnl),
      win: 'const WIN_ZIP_SHA256 = "da753f762bf2400e7191ec594086b186a7051d5af8dc886f6e2020c2403df738";',
      mac: 'const MAC_ARM64_TGZ_SHA256 = "65405dc8793c86cadb98b5e07f6d3bdde84f8300f1b030d4736b41c17610d6c1";',
      macComment: "// macOS (Microsoft publishes arm64 only for 1.24.2 / 1.25.0 — Apple Silicon)",
    };
    if (text.includes('const ORT_VERSION = "1.25.0";')) {
      info("fetch-onnxruntime.mjs: already on ONNX Runtime 1.25.0");
    } else if (text.includes(OLD.version) && text.includes(OLD.win) && text.includes(OLD.mac)) {
      let t = text.replace(OLD.version, () => NEW.version).replace(OLD.win, () => NEW.win).replace(OLD.mac, () => NEW.mac);
      if (t.includes(OLD.macComment)) t = t.replace(OLD.macComment, () => NEW.macComment);
      fetchPatched = t;
      info("fetch-onnxruntime.mjs: ONNX Runtime 1.24.2 -> 1.25.0 (checksums replaced)");
    } else {
      warn("scripts/fetch-onnxruntime.mjs does not match the known 1.24.2 content; ONNX Runtime was NOT bumped (edit ORT_VERSION and both SHA256 values by hand, see README).");
    }
  }
} else {
  info("fetch-onnxruntime.mjs: skipped (--skip-ort-bump)");
}

// ── write ──────────────────────────────────────────────────────────────────────────────────────────────────────────
if (dryRun) {
  info("--dry-run: nothing written.");
  process.exit(0);
}
if (patched !== null) {
  const backup = source + ".bak-before-parakeet-fix";
  if (!existsSync(backup)) copyFileSync(source, backup);
  writeFileSync(source, patched, "utf8");
  info(`updated:  ${source}\nbackup:   ${backup}`);
}
for (const m of modules) {
  const from = join(zipRoot, "src-tauri", "src", "stt", m);
  const to = join(sttDir, m);
  if (resolve(from) === resolve(to)) {
    info(`module:   ${to} (already in place)`);
  } else {
    copyFileSync(from, to);
    info(`module:   ${to}`);
  }
}
if (fetchPatched !== null) {
  writeFileSync(fetchScript, fetchPatched, "utf8");
  info(`updated:  ${fetchScript}`);
}
info("\nNext: cd src-tauri && cargo check   (then run the app; the dev log shows the 'NeMo Parakeet route selected' line)");
