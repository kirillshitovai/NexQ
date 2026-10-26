Test fixtures for tools/ort_harness (16 kHz mono PCM16).
ru_example.wav  test_wavs/example.wav of the sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19 model archive
ru_0.wav        test_wavs/0.wav of the sherpa-onnx-zipformer-ru-2024-09-18 model archive
ru_1.wav        test_wavs/1.wav of the sherpa-onnx-zipformer-ru-2024-09-18 model archive
Both archives are published by k2-fsa/sherpa-onnx (https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models).
expect.txt      "file stem|expected text" - the reference transcripts the patch was measured against.
