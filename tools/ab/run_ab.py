#!/usr/bin/env python3
"""A/B driver: real kaldi-native-fbank (knf_dump) -> real ONNX model (ort_ab) for several Fbank / TDT configurations.

  run_ab.py --model-dir D --knf-dump ./knf_dump --ort-ab ./ort_ab --work W  wav1.wav[=reference text] wav2.wav ...

A reference text (after '=') is optional; CER/WER are computed against it after lower-casing, yo->ye and punctuation removal.
Only the standard library is used (the WAV files must already be PCM16 mono 16 kHz).
"""
import argparse, json, re, subprocess, sys
from pathlib import Path

CONFIGS = {
    # name: (knf_dump args, description)
    "testonnx":  (["--window", "hann",  "--high-freq", "0",    "--log", "floor",    "--snip-edges", "1", "--pad-seconds", "2"],
                 "sherpa test_onnx.py: hann, 0..Nyquist, log(max), snip_edges=true, +2 s zero tail"),
    "release":   (["--window", "hann",  "--high-freq", "0",    "--log", "floor",    "--snip-edges", "0", "--pad-seconds", "0"],
                 "sherpa-onnx v1.13.8 release behaviour: hann, 0..Nyquist, log(max), snip_edges=false, no tail"),
    "master":    (["--window", "povey", "--high-freq", "-400", "--log", "additive", "--snip-edges", "0", "--pad-seconds", "0"],
                 "sherpa-onnx master source: povey, 0..7600, log(x+2^-24), snip_edges=false, no tail"),
    "hann_nopad":(["--window", "hann",  "--high-freq", "0",    "--log", "floor",    "--snip-edges", "1", "--pad-seconds", "0"],
                 "test_onnx.py without the 2 s tail (isolates the effect of padding)"),
}

def norm(s):
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return s.split()

def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

def cer(h, r):
    h, r = " ".join(norm(h)), " ".join(norm(r))
    return lev(h, r) / max(1, len(r))

def wer(h, r):
    h, r = norm(h), norm(r)
    return lev(h, r) / max(1, len(r))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--knf-dump", required=True)
    ap.add_argument("--ort-ab", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--tdt", default="sherpa,nemo")
    ap.add_argument("--json", help="write all results here")
    ap.add_argument("wavs", nargs="+")
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in a.wavs:
        wav, _, ref = item.partition("=")
        stem = Path(wav).stem
        for cname in a.configs.split(","):
            kargs, _desc = CONFIGS[cname]
            pref = str(work / f"{stem}.{cname}")
            info = json.loads(subprocess.check_output([a.knf_dump, "--wav", wav, "--out", pref] + kargs))
            for tdt in a.tdt.split(","):
                out = json.loads(subprocess.check_output([a.ort_ab, "decode", "--model-dir", a.model_dir,
                                                          "--features", pref + ".raw.f32", "--tdt", tdt, "--threads", "2"]))
                r = dict(wav=stem, config=cname, tdt=tdt, frames=out["feature_frames"], enc_frames=out["encoder_frames"],
                         tokens=out["num_tokens"], raw_mean=info["raw_mean_of_bin_means"], raw_std=info["raw_mean_of_bin_std"],
                         text=out["text"])
                if ref:
                    r["cer"], r["wer"] = cer(out["text"], ref), wer(out["text"], ref)
                rows.append(r)
                print(f"{stem:12} {cname:11} tdt={tdt:6} frames={r['frames']:5} enc={r['enc_frames']:4} tok={r['tokens']:3} "
                      f"mean={r['raw_mean']:7.3f} std={r['raw_std']:6.3f}"
                      + (f" CER={r['cer']*100:5.1f}% WER={r['wer']*100:5.1f}%" if ref else "") + f" | {out['text']}")
                sys.stdout.flush()
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    for cname in a.configs.split(","):
        for tdt in a.tdt.split(","):
            sel = [r for r in rows if r["config"] == cname and r["tdt"] == tdt and "cer" in r]
            if sel:
                print(f"AVG {cname:11} tdt={tdt:6} CER={sum(r['cer'] for r in sel)/len(sel)*100:5.1f}%  WER={sum(r['wer'] for r in sel)/len(sel)*100:5.1f}%  (n={len(sel)})")

if __name__ == "__main__":
    main()
