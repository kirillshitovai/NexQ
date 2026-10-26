#!/usr/bin/env bash
# Builds the reference tools of tools/ab on Linux x64 (g++ >= 9).
#
#   KNF_DIR   kaldi-native-fbank checkout   (tested: github.com/csukuangfj/kaldi-native-fbank, tag v1.22.4)
#   KISS_DIR  kissfft checkout              (tested: github.com/mborgerding/kissfft, commit febd4ca)
#   ORT_DIRS  space separated directories of unpacked ONNX Runtime releases, e.g.
#             "/opt/ort/onnxruntime-linux-x64-1.24.2 /opt/ort/onnxruntime-linux-x64-1.25.0"
#             (https://github.com/microsoft/onnxruntime/releases/download/vX.Y.Z/onnxruntime-linux-x64-X.Y.Z.tgz)
#   OUT       output directory (default ./build)
#
# Result: $OUT/knf_dump and one $OUT/ort_ab_<version> per ORT directory. All ort_ab binaries are compiled against the
# first directory's headers; the library each one loads is the one it was linked with (rpath), so the same source and
# the same model run on different ONNX Runtime versions - exactly what ort_versions.py compares.
set -euo pipefail
: "${KNF_DIR:?}" "${KISS_DIR:?}" "${ORT_DIRS:?}"
OUT="${OUT:-$PWD/build}"; HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT/obj"

gcc -O2 -c "$KISS_DIR/kiss_fft.c"  -I"$KISS_DIR" -o "$OUT/obj/kiss_fft.o"
gcc -O2 -c "$KISS_DIR/kiss_fftr.c" -I"$KISS_DIR" -o "$OUT/obj/kiss_fftr.o"
SRCS=$(ls "$KNF_DIR"/kaldi-native-fbank/csrc/*.cc | grep -v '/test-')
g++ -O2 -std=c++17 -I"$KNF_DIR" -I"$KISS_DIR" "$HERE/knf_dump.cpp" $SRCS "$OUT/obj/kiss_fft.o" "$OUT/obj/kiss_fftr.o" -o "$OUT/knf_dump"
echo "built $OUT/knf_dump"

FIRST=""
for d in $ORT_DIRS; do
  [ -n "$FIRST" ] || FIRST="$d"
  ver="${d##*-}"
  g++ -O2 -std=c++17 -I"$FIRST/include" "$HERE/ort_ab.cpp" -L"$d/lib" -lonnxruntime -Wl,-rpath,"$d/lib" -o "$OUT/ort_ab_$ver"
  echo "built $OUT/ort_ab_$ver  (links $d/lib)"
done
