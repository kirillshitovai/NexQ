#!/usr/bin/env python3
"""Short-segment sweep: cut windows out of real WAVs the way an RMS segmenter would (speech + ~0.8 s of quiet tail),
then decode every window with several Fbank configurations on the real model and count empty / differing results.

  seg_sweep.py --model-dir D --knf-dump ./knf_dump --ort-ab ./ort_ab --work W a.wav b.wav ...
Needs numpy (WAV slicing) in addition to the standard library.
"""
import argparse, json, subprocess, wave
from pathlib import Path
import numpy as np

def read(p):
    w = wave.open(str(p)); assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
    return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768

def write(p, x):
    w = wave.open(str(p), "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()); w.close()

CONFIGS = {
  "testonnx_pad2": ["--window","hann","--high-freq","0","--log","floor","--snip-edges","1","--pad-seconds","2"],
  "release_nopad": ["--window","hann","--high-freq","0","--log","floor","--snip-edges","0","--pad-seconds","0"],
  "hann_pad0_snip1": ["--window","hann","--high-freq","0","--log","floor","--snip-edges","1","--pad-seconds","0"],
  "release_pad0.5": ["--window","hann","--high-freq","0","--log","floor","--snip-edges","0","--pad-seconds","0.5"],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True); ap.add_argument("--knf-dump", required=True)
    ap.add_argument("--ort-ab", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--lens", default="1.0,2.0,3.5"); ap.add_argument("--hop", type=float, default=1.7)
    ap.add_argument("--tail", type=float, default=0.8, help="seconds of quiet (noise rms 0.001) appended to every window")
    ap.add_argument("--json"); ap.add_argument("wavs", nargs="+")
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    segs = []
    for wp in a.wavs:
        x = read(wp)
        for L in [float(v) for v in a.lens.split(",")]:
            n = int(L * 16000); hop = int(a.hop * 16000)
            for s in range(0, max(1, len(x) - n), hop):
                seg = x[s:s + n]
                if np.sqrt(np.mean(seg ** 2)) < 0.01:  # skip windows that are (nearly) silent
                    continue
                tail = rng.normal(0, 0.001, int(a.tail * 16000))
                segs.append((f"{Path(wp).stem}_{L:g}s_{s/16000:.1f}", np.concatenate([seg, tail])))
    print(f"{len(segs)} segments")
    results = {c: [] for c in CONFIGS}
    for name, x in segs:
        wavp = work / f"{name}.wav"; write(wavp, x)
        for c, kargs in CONFIGS.items():
            pref = str(work / f"{name}.{c}")
            subprocess.check_output([a.knf_dump, "--wav", str(wavp), "--out", pref] + kargs)
            results[c].append(pref + ".raw.f32")
    texts = {}
    for c, files in results.items():
        out = subprocess.check_output([a.ort_ab, "decode", "--model-dir", a.model_dir, "--features", ",".join(files), "--threads", "2"]).decode()
        texts[c] = [json.loads(l)["text"] for l in out.strip().splitlines()]
    base = texts["release_nopad"]
    print(f"{'config':18} empty  differs_from_release_nopad")
    for c in CONFIGS:
        empty = sum(1 for t in texts[c] if not t.strip())
        diff = sum(1 for t, b in zip(texts[c], base) if t.strip().lower() != b.strip().lower())
        print(f"{c:18} {empty:3}/{len(segs)}  {diff:3}/{len(segs)}")
    if a.json:
        Path(a.json).write_text(json.dumps({"names": [n for n, _ in segs], "texts": texts}, ensure_ascii=False, indent=1), encoding="utf-8")

if __name__ == "__main__":
    main()
