#!/usr/bin/env python3
"""Compare ONNX Runtime builds on phrase-like segments cut from real WAVs.

Why: the int8 Parakeet encoder gives DIFFERENT results for the SAME features on different ONNX Runtime versions
(measured: ORT <= 1.24.x returns an empty text for short, abruptly starting fragments that ORT >= 1.25 transcribes).
NexQ 2.26.0 ships 1.24.2; the Parakeet patch moves it to 1.25.0.

  ort_versions.py --wav-dir D --work W --knf-dump build/knf_dump --model M --runner label=path/to/ort_ab [--runner ...] [--abrupt]

Default mode: every WAV is cut at its quietest points into segments of ~1.5-5 s (so the cuts fall into pauses, like
the app's segmenter does), 0.2 s of faint noise is put in front (the segmenter's preroll) and 0.8 s behind (its
pause tail).
--abrupt: windows of 1.0 / 2.0 / 3.5 s are cut at arbitrary points (hop 1.7 s) and fed with NO padding - short,
abruptly starting fragments, the worst case for the encoder. The LAST --runner is the reference.
Features come from the real kaldi-native-fbank (knf_dump, release configuration), decoding from ort_ab; only the
ORT library differs between the runners. Prints one line per segment and a summary per runner.
"""
import argparse, json, math, os, random, struct, subprocess, sys, wave


def read_wav(p):
    w = wave.open(p)
    assert w.getframerate() == 16000 and w.getsampwidth() == 2
    n = w.getnframes()
    raw = w.readframes(n)
    ch = w.getnchannels()
    x = struct.unpack('<%dh' % (n * ch), raw)
    return list(x[::ch])


def write_wav(p, x):
    o = wave.open(p, 'wb')
    o.setnchannels(1); o.setsampwidth(2); o.setframerate(16000)
    o.writeframes(struct.pack('<%dh' % len(x), *x))
    o.close()


def block_rms(x, size=160):
    return [math.sqrt(sum((v / 32768) ** 2 for v in x[i:i + size]) / size) for i in range(0, len(x) - size + 1, size)]


def cut_points(x, lo=1.5, hi=5.0):
    """Quietest 100 ms window inside [lo, hi] seconds after the previous cut, repeatedly."""
    r = block_rms(x)
    cuts, start = [], 0
    while (len(r) - start) / 100 > hi + 0.5:
        a, b = start + int(lo * 100), start + int(hi * 100)
        best = min(range(a, min(b, len(r) - 10)), key=lambda i: sum(r[i:i + 10]))
        c = (best + 5) * 160
        cuts.append(c)
        start = best + 5
    return cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav-dir', required=True); ap.add_argument('--work', required=True)
    ap.add_argument('--knf-dump', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--runner', action='append', required=True, help='label=path of an ort_ab built against one ORT version')
    ap.add_argument('--threads', default='2')
    ap.add_argument('--abrupt', action='store_true', help='arbitrary windows without padding instead of cuts at pauses')
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    rnd = random.Random(1)
    quiet = lambda n: [int(rnd.uniform(-1, 1) * 0.0008 * 1.7320508 * 32768) for _ in range(n)]
    segs = []
    for f in sorted(os.listdir(a.wav_dir)):
        if not f.endswith('.wav'):
            continue
        x = read_wav(os.path.join(a.wav_dir, f))
        if a.abrupt:
            i = 0
            for length in (1.0, 2.0, 3.5):
                start = 0
                while start + int(length * 16000) <= len(x):
                    name = '%s_a%.1f_%02d' % (os.path.splitext(f)[0], length, i)
                    write_wav(os.path.join(a.work, name + '.wav'), x[start:start + int(length * 16000)])
                    segs.append((name, length))
                    start += int(1.7 * 16000)
                    i += 1
            continue
        bounds = [0] + cut_points(x) + [len(x)]
        for i in range(len(bounds) - 1):
            part = x[bounds[i]:bounds[i + 1]]
            if len(part) < 8000:
                continue
            name = '%s_%02d' % (os.path.splitext(f)[0], i)
            write_wav(os.path.join(a.work, name + '.wav'), quiet(3200) + part + quiet(12800))
            segs.append((name, len(part) / 16000))
    feats = []
    for name, _ in segs:
        out = os.path.join(a.work, 'f_' + name)
        subprocess.run([a.knf_dump, '--wav', os.path.join(a.work, name + '.wav'), '--out', out, '--window', 'hann', '--high-freq', '0',
                        '--log', 'floor', '--snip-edges', '0', '--pad-seconds', '0'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        feats.append(out + '.raw.f32')
    results = {}
    for r in a.runner:
        label, path = r.split('=', 1)
        p = subprocess.run([path, 'decode', '--model-dir', a.model, '--features', ','.join(feats), '--threads', a.threads], capture_output=True, text=True)
        results[label] = [json.loads(l) for l in p.stdout.splitlines() if l.startswith('{')]
        assert len(results[label]) == len(segs), (label, p.stderr[-300:])
    labels = list(results)
    print('%-14s %5s  %s' % ('segment', 'sec', '  '.join('%-7s' % l for l in labels)))
    for i, (name, sec) in enumerate(segs):
        print('%-14s %5.1f  %s  | %s' % (name, sec, '  '.join('%-7d' % results[l][i]['num_tokens'] for l in labels), results[labels[-1]][i]['text'][:60]))
    print()
    for l in labels:
        n_empty = sum(1 for r in results[l] if r['num_tokens'] == 0)
        print('%-8s segments=%d  empty=%d  tokens total=%d' % (l, len(segs), n_empty, sum(r['num_tokens'] for r in results[l])))
    ref = labels[-1]
    speech = [i for i in range(len(segs)) if results[ref][i]['num_tokens'] > 0]
    print('reference (%s) transcribes %d of %d segments (the rest is silence/noise or a fragment of a word)' % (ref, len(speech), len(segs)))
    for l in labels:
        print('%-8s empty on %d of those %d' % (l, sum(1 for i in speech if results[l][i]['num_tokens'] == 0), len(speech)))
    for l in labels[:-1]:
        same = sum(1 for i in range(len(segs)) if results[l][i]['text'] == results[ref][i]['text'])
        print('%s vs %s: identical text in %d/%d segments' % (l, ref, same, len(segs)))
    json.dump({'segments': segs, 'results': results}, open(os.path.join(a.work, 'result.json'), 'w'), ensure_ascii=False)


if __name__ == '__main__':
    main()
