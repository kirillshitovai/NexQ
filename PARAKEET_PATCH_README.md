# NexQ v2.26.0 — Parakeet TDT 0.6B v3 INT8, ревизия 3

Патч чинит пустую расшифровку русской речи моделью NeMo Parakeet TDT 0.6B v3 (INT8) в NexQ 2.26.0 (`encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt`). Поток данных: `OrtStreamingSTT::feed_audio` → `AudioMessage::Samples(Vec<i16>)` → поток `ort-stt-inference` → `TranscriptResult` в `result_tx`; микрофон («You») и системный звук («Them») идут двумя независимыми экземплярами этого провайдера.

> **Что НЕ доказано.** Я не запускал приложение под Windows и не слышал ваш микрофон. Причина вашей пустой расшифровки по-прежнему гипотеза (см. раздел 3). Патч сделан так, чтобы это выяснилось за один запуск: подробный лог, жёсткие ошибки вместо тишины и workflow, который гоняет настоящую модель на Windows.

## 1. Установка

**Вариант А — просто распаковать ZIP поверх корня NexQ** (пути в ZIP относительные от корня репозитория). Файл `ort_streaming.rs` в ZIP — это v2.26.0 с двумя вставками (проверено: побайтово совпадает с результатом установщика на чистом дереве).

**Вариант Б — если в вашем `ort_streaming.rs` есть свои правки или он уже патчен ревизией 1/2:** распакуйте ZIP в любую папку и выполните оттуда

```
node parakeet-patch/apply-patch.mjs --project-root C:\путь\к\NexQ        # или parakeet-patch\apply-patch.ps1 -ProjectRoot ...
```

Установщик копирует два модуля, добавляет в `ort_streaming.rs` две строки `#[path] mod …` и ветку `if … contains("parakeet")`, при этом апгрейдит деревья ревизий 1/2, повторный запуск ничего не ломает, чужой/частично патченный файл отвергает без изменений (бэкап `ort_streaming.rs.bak-before-parakeet-fix`). `--dry-run` показывает, что будет сделано; `--skip-ort-bump` оставляет ONNX Runtime 1.24.2. Вручную то же делает `git apply parakeet-patch/ort_streaming.integration.patch`.

Дальше: `cd src-tauri && cargo check`, обычная сборка в Actions. Новый workflow `.github/workflows/parakeet-check.yml` — отдельный, `ci.yml` не тронут.

## 2. Найденные ошибки

| № | Причина | Файл (v2.26.0) | Функция | Что сделано |
|---|---|---|---|---|
| 1 | Вход энкодера собирался как `[1, T, C]`, у NeMo-модели он `[1, 128, T]`; выход читался как `[1, T, C]`, а он `[1, 1024, T']` | `ort_streaming.rs` | `EncoderState::run_encoder` (стр. ~457) | `OrtBackend::encode`: `[1,128,T]` + `length` int64; выход проверяется как `[1,1024,T']` |
| 2 | Вход joiner шёл как `[1,1,dim]`/`[1,dim]`, у NeMo `[1,1024,1]` и `[1,640,1]` | `ort_streaming.rs` | `run_joiner` (стр. ~1272) | `OrtBackend::joiner` |
| 3 | `BLANK_ID = 0`; у Parakeet v3 blank = 8192 (`<blk>`), токенов 8193 | `ort_streaming.rs` | `const BLANK_ID`, `detokenize` | `Vocab::parse_strict`: ищет `<blk>`, проверяет размер словаря, нумерацию подряд, дубликаты, обязательный blank в конце. Любое несовпадение — ошибка старта |
| 4 | TDT декодировался как обычный transducer: argmax по всем 8198 логитам (8193 токена + 5 длительностей) | `ort_streaming.rs` | цикл greedy decode (стр. ~830) | `transcribe`: токен — argmax первых 8193, длительность — argmax последних 5, правило как в sherpa-onnx `DecodeOneTDT` (blank при нулевой длительности → шаг 1; состояние декодера как в C++) |
| 5 | Streaming-архитектура Zipformer (блоки `CHUNK_SAMPLES` по 320 мс с состоянием энкодера, граница фразы после `BLANK_THRESHOLD = 75` подряд blank-кадров ≈ 3 с) применена к офлайн-модели, которая обрабатывает фразу целиком и не имеет streaming-состояния | `ort_streaming.rs` | `inference_thread_main` | Ветка `parakeet` → `nemo_parakeet::run`: RMS-сегментатор + офлайн-распознавание сегмента; готовая фраза уходит после паузы 0,8 с, хвост — при остановке |
| 6 | Front end `FbankExtractor` — HTK-mel (2595·log10) без Slaney-нормировки и без per-feature нормализации; `normalize_type=per_feature` из метаданных энкодера не учитывался | `stt/fbank.rs` | `build_mel_filterbank`, `process_chunk` | Собственный Fbank в `nemo_parakeet_core.rs`: Slaney-mel, преэмфазис 0,97, Hann, per-feature нормализация. Сверен с настоящим kaldi-native-fbank (раздел 4) |
| 7 | Не проверялось, что сессии соответствуют ожидаемому контракту; ошибка формы всплывала как пустой текст | — | — | `load_model`: типы/ранги/статические размеры входов и выходов всех трёх сессий, метаданные (`pred_rnn_layers`, `pred_hidden`, `normalize_type`), число логитов joiner = токены + 5, самотест на 1 с тихого шума через все три сессии. Всё — жёсткие ошибки с текстом в Dev Log |
| 8 | ONNX Runtime 1.24.2: INT8-энкодер возвращает пустой текст на коротких «резко начинающихся» фрагментах и хуже переносит рецепт Fbank (раздел 3) | `scripts/fetch-onnxruntime.mjs` | `ORT_VERSION`, SHA256 | 1.25.0, хэши проверены по скачанным архивам; на известной сборке 1.24.x автоматически включается более устойчивый рецепт `test_onnx` |
| 9 | Ни одной диагностики в потоке: не видно, дошли ли отсчёты, создан ли сегмент, что вернул joiner, ушёл ли `TranscriptResult` | — | — | Лог каждого этапа (раздел 5), ошибки `blocking_send` не глотаются; паника потока логируется (`catch_unwind`) |
| 10 | Принудительный разрез длинной речи на 10 с мог попасть в середину слова | (моя же ревизия 2) | `Segmenter` | Разрез по самому тихому окну 100 мс из последних 2 с; остаток переносится в следующий сегмент без потери отсчётов |
| 11 | При остановке потока хвост очереди терялся: `stop_stream` ставит флаг раньше, чем поток разберёт уже отправленное аудио | (моя же ревизия 2) | `drive` | При флаге остановки поток дочитывает очередь до `Stop`/пустой очереди (лимит 60 с), потом сбрасывает открытый сегмент. Тест на этом дереве падает на старом поведении (проверено мутацией) |

## 3. Что показали измерения

Эталон — настоящая модель из `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2`, настоящий `kaldi-native-fbank` v1.22.4, референс `sherpa-onnx-offline` v1.13.8. Данные: 7 файлов (de/en/es/fr из архива модели, 3 русских из релизов sherpa-onnx; русские лежат в `tools/ort_harness/testdata`), CER после нормализации регистра/пунктуации.

**Рецепт Fbank × версия ONNX Runtime** (средний CER на 7 файлах, правило TDT как в sherpa):

| рецепт | ORT 1.24.2 | ORT 1.25.0 | ORT 1.28.0 |
|---|---|---|---|
| `release` — продакшн sherpa: Hann, `snip_edges=false`, без добавки тишины | 6,3 % | **1,1 %** | 1,0 % |
| `test_onnx` — `test_onnx.py`: `snip_edges=true`, +2 с тишины | **1,3 %** | 1,3 % | 1,3 % |
| `hann_nopad` — то же без добавки | 6,2 % | 1,3 % | 1,6 % |
| `master` — Povey, `high_freq=-400`, аддитивный лог | 19,3 % | 15,9 % | 14,2 % |

Правило TDT: на ORT ≥ 1.25 sherpa-правило и «чистое NeMo» дают одинаковый CER; на 1.24.2 с `release` NeMo-правило хуже (23,7 % против 6,3 %). Поэтому по умолчанию — sherpa-правило, как в эталонном C++; `MAX_SYMBOLS_PER_FRAME=10` не используется. Рецепт `master` из репозитория sherpa заметно хуже во всех версиях и не используется.

**Короткие фрагменты** (`tools/ab/ort_versions.py --abrupt`: 63 окна по 1/2/3,5 с, без добавок; 52 из них эталон расшифровывает):

| ORT | пусто на окнах с речью |
|---|---|
| 1.24.2 | 17 из 52 |
| 1.25.0 | 1 из 52 |
| 1.28.0 (эталон) | 0 из 52 |

**Сегменты, разрезанные по паузам** (как делает сегментатор приложения, 0,2 с преролла и 0,8 с хвоста; 12 сегментов): пустых на речи нет ни в одной версии (8/12 и 10/12 текстов идентичны эталону для 1.24.2 и 1.25.0).

Вывод: ORT 1.24.2 даёт пустой текст на коротких фрагментах с резким началом и теряет слова при рецепте без добавки; ≥ 1.25.0 — нет. Для обычных фраз, вырезанных по паузам, разницы почти нет. Это **объясняет часть** сценариев «пусто», но **не доказано**, что это причина именно у вас: Windows-сборку ORT 1.25.0 я не запускал, а до этого патча в коде были ошибки 1–5, каждая сама по себе давала пустой/мусорный текст. Как проверить на Windows — раздел 6.

## 4. Проверки Fbank и цикла TDT

* Собственный Fbank (на `std`, свой FFT) против настоящего kaldi-native-fbank на 7 WAV × 4 конфигурации (28 пар): сырой log-mel `max|Δ| ≤ 4,2e-4`, `mean|Δ| ≤ 1e-6`; после нормализации `max|Δ| ≤ 1,4e-3` (рецепт `master`), для `release` ≤ 8,1e-4. Это совпадение с эталонной реализацией, а не «точная копия NeMo»: NeMo-препроцессор на PyTorch я не запускал.
* Rust-цикл TDT с настоящей моделью даёт те же тексты, что C++-цикл из sherpa-onnx.
* Полный путь `drive` (сегментатор → Fbank → энкодер → TDT → `Emission`) на реальном потоке из 7 фраз с паузами 1,2 с, при размерах чанка 320 / 441 / 4096 отсчётов: ORT 1.25.0 — 8 результатов, CER 1,7 %; ORT 1.24.2 (автовыбор `test_onnx`) — 8 результатов, CER 2,4 %; текст не зависит от размера чанка.

## 5. Что читать в Dev Log

Источник `ort_streaming`. Для одного запуска вы должны увидеть по порядку:

```
NeMo Parakeet route selected (revision 3). files: encoder=encoder.int8.onnx decoder=… joiner=… tokens=tokens.txt
tokens.txt OK: 8193 tokens, blank_id=8192 ("<blk>")
ONNX Runtime library loaded (ensure_loaded OK); build info: <строка сборки>
Encoder loaded: inputs=[audio_signal F32[-1, 128, -1], length I64[-1]] outputs=[…]      (то же для Decoder, Joiner)
model contract OK: encoder [1,128,T]->[1,1024,T'], decoder states [2,1,640], joiner logits = 8193 tokens + 5 durations, normalize_type="per_feature"
frontend: fbank=release (…)            decoder loop: tdt=sherpa (zero skip -> 1)
segmenter (RMS, not a VAD): start>0.0040 continue>0.0025 pause=800ms max=10000ms … min=300ms preroll=200ms
self-test OK in … ms: 100 feature frames -> 13 encoder frames, 10 joiner calls, 0 tokens
NeMo Parakeet ready; waiting for audio
first audio chunk reached the NeMo thread: N samples (expected 16 kHz mono i16)
audio last 10s: … samples = 16000 Hz (16000 expected), rms … dBFS, peak … dBFS | … segments started=… | results sent=… send_errors=0     (каждые 10 с)
segment started at 0.85s of stream (block rms 0.2405 = -12.4 dBFS)
segment ort_7 ended (Pause): 3.63s starting at 0.85s, decoding
segment ort_7: 363 feature frames -> 46 encoder frames, 18 joiner calls, 16 tokens (0 control skipped), durations [..], raw log-mel min=… max=… mean=…, 611 ms; text: …
segment ort_7: TranscriptResult sent
```

Как читать сбои:

| что в логе | значит |
|---|---|
| нет строки `route selected` | ветка не подключена: `ort_streaming.rs` не патчен, или путь модели не содержит `parakeet` |
| `tokens.txt rejected` / `Encoder … is …, expected …` / `Joiner outputs N logits …` / `self-test failed` | модель или `tokens.txt` не те, что ожидалось; транскрипции не будет, причина написана |
| `ONNX Runtime library: …` | `onnxruntime.dll` не найден или не загрузился |
| `this is an official ONNX Runtime 1.24.x build` (warn) | собранное приложение всё ещё на 1.24.2: поднимите ORT до 1.25.0 |
| нет `first audio chunk` | отсчёты не доходят до потока (проблема выше: захват, `feed_audio`) |
| `audio arrives at N samples/s, not 16000` | неверная частота после ресэмплера или потеря чанков |
| `audio … is digital silence (peak = 0)` | захват отдаёт нули: устройство, mute |
| в строке статистики `blocks above start` = 0 при говорении | речь ниже порога сегментатора (0,004 RMS ≈ −48 dBFS) |
| `segment … model returned NO text` (warn) | сегмент дошёл до модели и вернул 0 токенов: сравните `raw log-mel`, длину, ORT |
| `TranscriptResult NOT delivered` (error) | `blocking_send` не удался: получатель закрыт |
| `NeMo thread panicked` | паника, текст паники в логе |

`NEXQ_PARAKEET_DUMP_DIR=C:\temp\pk` — каждый сегмент, отправленный в модель, дополнительно пишется как WAV 16 кГц: его можно прослушать и прогнать через sherpa-onnx.

## 6. Проверка на Windows одним кликом

Actions → **Parakeet check** → *Run workflow*. Это и есть замена запуску приложения, который я сделать не могу:

* `core-tests` — модульные тесты на Windows-раннере (35 штук, только `std`);
* `harness` — собирает `tools/ort_harness`: **настоящие** `nemo_parakeet*.rs` + **настоящий** крейт `ort` (те же features, что у NexQ, версии из вашего `Cargo.lock`) — это и есть проверка компиляции Rust-обвязки с реальным API;
* `real-model` (только ручной запуск) — скачивает модель (тот же архив, что качает NexQ), берёт официальный `onnxruntime.dll` выбранной версии (1.25.0 или 1.24.2; SHA256 сверяется), прогоняет три русских WAV через `nemo_parakeet::run` теми же каналами, что в приложении, и проверяет: результаты пришли, `is_final`, уникальные `segment_id`, нет `[error]`, CER потока ≤ 10 %, текст не зависит от размера чанка, битый `tokens.txt` даёт жёсткую ошибку. Поле *variant* позволяет форсировать рецепт: запуск на 1.24.2 с `release` против 1.25.0 покажет на вашей платформе, действительно ли дело в версии ORT.

Это всё ещё не микрофон и не Tauri: реальный захват, ресэмплер и UI проверяются только запуском приложения (раздел 8).

## 7. Переменные окружения (читаются при старте потока)

| переменная | значения | по умолчанию |
|---|---|---|
| `NEXQ_PARAKEET_VARIANT` | `release`, `test_onnx`, `master`, `hann_nopad` | `release`; на известной сборке ORT 1.24.x — `test_onnx` |
| `NEXQ_PARAKEET_TDT` | `sherpa`, `nemo` | `sherpa` |
| `NEXQ_PARAKEET_DUMP_DIR` | папка | выключено |

Пустое значение = не задано; неизвестное значение — предупреждение в логе и значение по умолчанию.

## 8. Ограничения и границы проверки

* **Сегментатор — это RMS-порог, а не VAD.** Пороги: старт 0,004, продолжение 0,0025, пауза 0,8 с, максимум 10 с, минимум 0,3 с (включая преролл 0,2 с), порог не менялся без теста-кандидата. Проверено на реальной модели: речь на −30 дБ ловится, шум ниже порога игнорируется, речь, обрезанная на остановке, выдаётся, две фразы с паузой 1,2 с дают два результата. **Известное ограничение:** постоянный шум/гул выше порога (Zoom/Телемост с фоном, вентилятор) не даёт пауз: фразы будут выходить только по длине 10 с (разрез по самому тихому месту) или при остановке. Адаптивный шумовой порог или настоящий VAD — отдельная работа, тестов-кандидатов у меня нет.
* Промежуточных (is_final=false) результатов нет, как и было задумано: готовая фраза приходит после паузы.
* Тестовые данные короткие и чистые (дикторские записи); на разговорной речи через Zoom качество и поведение сегментатора могут отличаться.
* Модель `parakeet-tdt-ctc-110m` (CTC, без encoder/decoder/joiner) этим путём не идёт и не затронута.
* Подъём ONNX Runtime до 1.25.0 затрагивает и остальные локальные ONNX-движки NexQ; ветка `ort_streaming` Zipformer с 1.25.0 мной не запускалась. Крейт `ort 2.0.0-rc.12` собран под API 1.24 и на более новую минорную версию библиотеки лишь пишет информационное сообщение (по исходникам крейта); workflow `real-model` проверяет именно эту пару на Windows.

## 9. Статус

**Скомпилировалось**
* `nemo_parakeet_core.rs` — `rustc --edition 2021 --test -O`, 35 тестов проходят, 4 реальных теста помечены `#[ignore]`.
* Утилиты `tools/ab` (`knf_dump`, `ort_ab`) — g++ на Linux, сборка с нуля скриптом `build.sh`.
* Установщик `apply-patch.mjs` и `.patch`: проверены на копии дерева v2.26.0 (чистое дерево, повторный запуск, дерево ревизии 1/2, CRLF, чужая версия, частично патченный файл, `--dry-run`); `patch -p1` и `git apply --check` проходят.

**Только проверка типов по рукописным заглушкам (это НЕ `cargo check`)**
* `nemo_parakeet.rs` и `tools/ort_harness` — крейты `ort`/`ndarray`/`tokio` из sandbox недоступны (crates.io заблокирован), поэтому сигнатуры были выписаны вручную по исходникам `ort 2.0.0-rc.12` и обвязка проверялась против них. Реальную компиляцию выполнят `ci.yml` (`cargo check`) и job `harness`.

**ONNX-тест пройден** (Linux, настоящая модель, настоящий ORT; Rust-front end, Rust-цикл TDT и Rust-`drive`, графы исполняет референс-процесс `ort_ab`)
* сравнение Fbank с kaldi-native-fbank; A/B рецептов и версий ORT; фрагменты и сегменты; сценарии сегментатора; поток из 7 фраз при трёх размерах чанка на ORT 1.25.0 и на 1.24.2 (раздел 4).
* Графы исполнялись C++-обёрткой с версиями ORT 1.24.1/1.24.2/1.25.0/1.26.0/1.27.0/1.28.0 и 1.28.2 (из sherpa-onnx); вызовы через крейт `ort` — нет.

**В приложении проверено** — ничего. Ни Tauri-сборка, ни запуск, ни микрофон/loopback.

**Не проверено**
* компиляция Rust-обвязки с настоящим `ort` и сборка Tauri на Windows (проверит CI);
* поведение ORT 1.25.0 на Windows и на вашем процессоре (проверит `real-model`);
* реальный захват микрофона и системного звука, ресэмплер, UI-транскрипт обеих сторон;
* что ваша пустая расшифровка объясняется именно ORT 1.24.2 (а не ошибками 1–5 или входным аудио);
* Zipformer и другие ONNX-функции NexQ на ORT 1.25.0;
* долгая работа (утечки памяти, несколько часов сессии).

Поправка к промежуточным результатам этой сессии: числа CER и вывод «добавка 2 с тишины даёт пустой результат на коротких фразах», если вы их видели, сняты — они получены на неучтённой тогда, более новой версии ORT. Здесь версия всегда указана явно: на 1.24.2 добавка, наоборот, самый устойчивый рецепт, а на ≥ 1.25 лучше рецепт без добавки.

## 10. Воспроизведение измерений (Linux)

```
# сборка эталонных утилит (нужны исходники kaldi-native-fbank v1.22.4, kissfft febd4ca и распакованные ORT)
KNF_DIR=… KISS_DIR=… ORT_DIRS="/opt/ort/onnxruntime-linux-x64-1.24.2 /opt/ort/onnxruntime-linux-x64-1.25.0" tools/ab/build.sh

# A/B рецептов и правил TDT на целых файлах
python3 -I tools/ab/run_ab.py --model-dir MODEL --knf-dump build/knf_dump --ort-ab build/ort_ab_1.25.0 --work W a.wav b.wav …

# версии ORT на коротких фрагментах / на сегментах по паузам
python3 -I tools/ab/ort_versions.py --wav-dir WAVS --work W --knf-dump build/knf_dump --model MODEL \
        --runner ort1.24.2=build/ort_ab_1.24.2 --runner ort1.25.0=build/ort_ab_1.25.0 --runner ort1.28.0=build/ort_ab_1.28.0 [--abrupt]

# модульные и реальные тесты ядра
rustc --edition 2021 --test -O src-tauri/src/stt/nemo_parakeet_core.rs -o core_tests && ./core_tests
NEXQ_ORT_AB=build/ort_ab_1.25.0 NEXQ_MODEL_DIR=MODEL NEXQ_WAV_DIR=WAVS NEXQ_EXPECT_FILE=expect.txt NEXQ_MAX_CER=0.1 \
   ./core_tests --ignored real_driver_stream --nocapture        # также real_model_transcription, real_segmenter_scenarios,
                                                                 # real_knf_feature_comparison (нужен NEXQ_KNF_DIR)
```

## 11. Состав ZIP

```
PARAKEET_PATCH_README.md
parakeet-patch/apply-patch.mjs, apply-patch.ps1, ort_streaming.integration.patch
src-tauri/src/stt/nemo_parakeet.rs          (ONNX-обвязка: проверка контракта, Backend на ort, запуск потока)
src-tauri/src/stt/nemo_parakeet_core.rs     (Fbank, TDT, tokens.txt, сегментатор, цикл приёма; только std)
src-tauri/src/stt/ort_streaming.rs          (v2.26.0 + 2 строки mod + ветка parakeet)
scripts/fetch-onnxruntime.mjs               (ORT 1.25.0)
.github/workflows/parakeet-check.yml
tools/ort_harness/                          (проверка на настоящих ort + модели; testdata — 3 русских WAV)
tools/ab/                                   (эталонные утилиты и скрипты измерений)
```
