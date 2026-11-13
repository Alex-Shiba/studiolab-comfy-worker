# Лаборатория H3 для A/B на 5090

## vsr (lab)

`COMFY_VSR=1` добавляет увеличение готового MP4 через ReCaVSR. Без точного значения `1`
режим неизвестен, пакет и веса не устанавливаются, команда запуска прежняя.
Можно включить вместе с `COMFY_REGEN=1`: устанавливаются и разрешаются оба набора узлов.

```json
{
  "request": "0123456789abcdef0123456789abcdef",
  "mode": "vsr",
  "video": "<base64 MP4 без data:-префикса>",
  "vsr": {"scale": 4, "fit": "2k"}
}
```

Допустимы только эти ключи; `fit` можно опустить. Нет внешних prompt/seed.
Источник: до 48 MiB, 2–15 с, один видеопоток, постоянные отметки кадров, длинная сторона ≤1920,
не более 60 fps и 900 кадров. Сначала проверяются метаданные без декодирования: число кадров,
длительность, плотность кадров, размеры и кодеки всех аудиопотоков. Неизвестное число кадров
отклоняется. Затем проверяются реальные отметки кадров с пределом чтения 901 видеопакет;
лишний кадр или несовпадение с числом кадров контейнера отклоняются до GPU.
FPS сохраняется, включая дробный; проверки не пересчитывают видео в 24 fps.
`scale` — целое 2 или 4. Площадь **до fit** ограничена 5376×3072; например, 1920×1080 при ×4
отклоняется даже с `fit`. `fit=2k|4k` только уменьшает до длинной стороны 2560|3840,
с сохранением пропорций и округлением до чётных размеров. Один проход ffmpeg/Lanczos, H.264 CRF 14;
без уменьшения видеопоток копируется. Исходные аудиопотоки всегда копируются без перекодирования,
включая задержку относительно видео и аудиохвост. Немой источник остаётся немым.
Принимаются только копируемые в MP4 аудиокодеки: `aac`, `mp3`, `opus`, `alac`, `ac3`, `eac3`;
PCM и другие кодеки отклоняются до декодирования видео.
Размеры, число кадров, дробная частота и длительность проверяются до и после mux.
Таймауты завершения зависят от числа кадров и площади: проверка/mux — минимум 60 с
или объём видео при 8 млн пикселей/с; fit — минимум 600 с или объём модельного выхода
при 1 млн пикселей/с. Эти консервативные бюджеты ещё требуют замера на живой машине.

Первая задача после READY начинает общую загрузку: `queued / downloading_vsr_models`, progress — доля байтов.
Остальные режимы не ждут её. Затем `running`, `finishing`, `done` или короткий `failed`;
новый request после сбоя загрузки повторяет попытку. State содержит `vsr: {scale,fit,width,height}`,
`width/height/frames`, числовой `fps`, точный рациональный `frame_rate` и `exec_s` без загрузки/очереди,
но с финальным mux. `model_s` — интервал WS-событий выполнения `ReCaVSRUpscale`, включая его decode/encode;
это не измерение CUDA-ядер. Без этих событий поле отсутствует. Промежуточный UUID MP4 пакета удаляется
из приватного temp после завершения; история, исходник и лишние файлы задания очищаются прежним способом.
45 минут на ComfyUI, exam, самоудаление и TTL результата не изменены.

Проверено чтением исходников и метаданных, без запуска пакета и загрузки весов:

- `ComfyUI-ReCaVSR`: `831769467c89b35ee47926189ac8189be63883d1` (HEAD на 2026-11-13).
  Шаблон `recavsr_video_4x_api.json` и `nodes.py`: `ReCaVSRLoader(transformer)`;
  `ReCaVSRUpscale(model,video,scale,color_fix,seed,decoder,kernels,kv_offload,encoder)`.
  Выбраны int8 ConvRot, FlashDecoder, `color_fix=none`, внутренний `seed=42`, `kv_offload=false`, `encoder=auto`.
  У опубликованного API-шаблона transformer — bf16; int8 здесь выбран явно.
  Вход `kernels` задаёт ускорение: install проверяет GPU в ComfyUI venv и выбирает `fast` при CC ≥8.0,
  `exact` при CC 7.5; более старые GPU не проходят установку VSR.
- В штатном ComfyUI commit `2c75dc1b379a59e0880619d029607af5169a62ef` проверены
  `LoadVideo(file)` и `SaveVideo(video,filename_prefix,format=mp4,format.codec=auto)`.
  Узел модели использует ID `16` для измерения, сохранение — `20`. Непроверенных входов графа нет.
- `recavsr/video_io.py`: VIDEO сохраняет рациональную частоту, но звук **перекодируется в AAC**;
  поэтому worker заменяет его исходными пакетами. `loader.py` требует три соседних файла
  в `models/recavsr`; Wan VAE не нужен.
- `requirements.txt` содержит только `triton` и `opencv-python-headless`.
  Только для VSR к `runtime-constraints.txt` добавлен `vsr-constraints.txt`:
  **triton 3.8.0**, **opencv-python-headless 5.0.0.93**. Метаданные Linux-wheel torch 2.14.1+cu130
  требуют `triton~=3.8.0`; на дату проверки в ветке 3.8 опубликован только 3.8.0.
  OpenCV предоставляет Linux x86-64 wheel `cp37-abi3`, совместимый с CPython 3.13;
  его `numpy>=2` допускает закреплённый numpy 2.5.4. Источники — wheel METADATA PyTorch
  и JSON PyPI, точные ссылки сохранены в `vsr-evidence.json`.
  Установка выполняется только в ComfyUI venv. Версии проверяются до и после установки;
  оставшееся несовпадение завершает стадию install с явной ошибкой. Требования regen прежние.
- В `vsr_files` ровно три файла, **6 499 853 216 байт**. Transformer: `PedroMarinhoDev/ReCaVSR-ComfyUI`,
  revision `2fffebd4413433ad4b8be7aec44b7fb60c897d6a`, **6 266 935 048 байт**.
  FlashDecoder и prompt фактически находятся в `kopper/ReCaVSR`, revision
  `9c874580eedeea516f31ee1f7c562a311dbf9cf7`: **228 723 776** и **4 194 392 байта**.
  Полные пути, SHA-256 из Hub LFS и commit пакета — [model-manifest.json](model-manifest.json);
  ссылки на проверенные источники — [vsr-evidence.json](vsr-evidence.json).

Чек-лист первого живого запуска (`needs`):

- Установка требований на Linux/Python 3.13.9, torch 2.14.1+cu130, разрешение зависимостей Triton/OpenCV;
  регистрация узлов вместе с regen и загрузка всех трёх закреплённых весов.
- INT8 и `fast` на выбранной GPU; fallback `exact` на CC 7.5, время первой компиляции Triton;
  наличие NVENC и переход на CPU-кодек, работа после regen с его `--lowvram`.
- Настоящие ×2/×4, 2/15 секунд, портрет/пейзаж и предел площади: VRAM/RAM/время,
  отсутствие потери кадров, точные FPS/длительность, хеши аудиопакетов и fit 2K/4K.
  Проверить бюджеты завершающих операций на пределе 900 кадров / 60 fps и площади выхода.
- Сохранность лица, движения губ, деталей и цветов; поведение после смены сцены.
  Качество и тождественность лица не доказываются тестами контракта.
- WS-измерение `model_s`; отсутствие оставшихся временных файлов, успешная следующая задача
  после реального CUDA OOM и чередование shooting/regen/vsr.

GPU, Vast и платные сервисы не запускались. Opus-review выполняет внешний процесс SuperAI.

<a id="regen"></a>

## Только refine: решение владельца, 2026-11-11

После сравнения Kling 720p и Seedance 480p (IC 1080p 5 с и refine 1080p 10 с) владелец решил:
«слушай ну я думаю что надо брать рефайн, он неплох, а 5 секунд в IC - ну честно говоря беполезно».
Refine работает на отдельном Vast endpoint `studiolab-regen-lab`: «Отдельная машина».
Режим включается только `COMFY_REGEN=1`. `regen.method` можно опустить или указать `refine`;
остальные значения отклоняются с `regen method must be refine`.
Черновик: 5–15 с, 120–360 кадров при 24 fps; результат: 1080p/2K.
`start_sigma`: 0,5–0,95, по умолчанию 0,8. Промпт передаётся без переписывания.

### Удалено 2026-11-11

Причина — решение владельца выше: отдельный пятисекундный метод больше не нужен.
Удалены IC-граф и два пакета узлов:
`slmonker/ComfyUI-OpenH3Regenerate2KIC` (commit `a85bd9664a752637c5ec88bf4b1930b4688fc251`),
`ComfyUI-LynnReal` из `LynnReal-AI/LynnReal-Omni` (commit `ad16ac787430bd22456ee55f018764cf53c25935`).
Размеры самих git-репозиториев в манифесте не зафиксированы.

| Удалённый файл | Байты |
| --- | ---: |
| IC LoRA: `loras/minimax_h3_open_regenerate_2k_ic_comfy_bf16.safetensors` | 5 868 430 560 |
| HyperFlow: `loras/minimax_h3_hyperflow_8step_v1.0_comfyui_pruned_bf16.safetensors` | 3 912 313 312 |
| LMS Sharp: `loras/minimax_h3_lms_v1.0_r64.safetensors` | 1 239 664 536 |
| Causal fast VAE v2: `vae/fast_h3_vae_v2_24l.safetensors` | 6 471 256 328 |
| Silence-латент удалённого пакета: `assets/silence_audio_124f.safetensors` | 26 768 |

Экономия загрузки весов при холодном старте: **17 491 664 736 байт** (около 17,5 GB),
плюс два git-пакета, включая указанный silence-латент. Refine кодирует молчание через audio VAE.

### Сохранённый граф

Refine: `OpenH3RegenPrepareDraft` (768p, все разрешённые кадры, 24 fps, паддинг до `17k+5`) → VAE encode →
`OpenH3RegenUpscale` с LBH 3D → `OpenH3RegenRefineLatent` с кодированным молчанием → три шага Euler, CFG 1.
База — прежняя FL2VA INT8. `OpenH3RegenLora` применяет **полный NVIDIA rank-576 merge** и все Sharp curve
branches, strength=curve_strength=1, факторы в BF16; таблица берётся из существующей Ref2VA INT8.
`CLIPTextEncode` получает исходный промпт. SOL повторяет опубликованный Draft Video workflow:
tau=1,2, start=0, end=0,9, dense blocks `0,1,49`, min_tokens=12288, extra_tokens=256,
`sink_conditioning=exact_kv_and_rows`; выключен только диагностический verbose.
Штатная release-сетка сборки — `0.909375, 0.857142857, 0.666666667, 0`.
По brief выбран `start_sigma=0.8`; нода вычисляет **shift-6** сетку `0.8, 0.666666667, 0.444444444, 0`.
Переданные значения 0,5–0,95 используют тот же алгоритм. Sigma-shift модели поверх него не добавляется.

Sampler остаётся узлом `16`, SaveVideo — `20`. Перед `/prompt` проверяются все классы графа
и скрытая зависимость `MinimaxH3LatentUpscaler3D`.
`LTXVSeparateAVLatent` отделяет видеотензор перед fast decode и tiled fallback.
`OpenH3RegenFinish` отсекает паддинг и центрирует 1088→1080; вертикальная ориентация сохраняется.
После SaveVideo ffmpeg копирует видеопоток и исходные сжатые аудиопакеты, ограничивая длительность
исходником. Немой черновик остаётся немым. Внешние протоколы при проверке и mux отключены.

### Узлы, файлы и загрузка

Пины и SHA-256 сохранены в [model-manifest.json](model-manifest.json), источники —
в [regen-evidence.json](regen-evidence.json). Whitelist и установка берут два пакета из `regen_nodes`:

- `goofyrodent/h3regen`, `9b686c5ffd380e9d1351f9b8beef6d20d703ca3e`: `OpenH3RegenPrepareDraft`,
  `OpenH3RegenLora`, `OpenH3RegenUpscale`, `OpenH3RegenRefineLatent`, `OpenH3RegenSigmas`,
  `OpenH3RegenFastVAEDecode`, `OpenH3RegenFinish`.
- `LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler`, `f34cadb19f10fee7e36091b690cac39b4eb40573`:
  `MinimaxH3LatentUpscaler3D`, вызываемый из `OpenH3RegenUpscale`.
- Штатный `Comfy-Org/ComfyUI` v0.44.0, `2c75dc1b379a59e0880619d029607af5169a62ef`:
  `UNETLoader`, `CLIPLoader`, `VAELoader`, `VAEEncode`, `CLIPTextEncode` (`nodes.py`);
  `BasicGuider`, `RandomNoise`, `KSamplerSelect`, `SamplerCustomAdvanced` (`nodes_custom_sampler.py`);
  `LoadVideo`, `GetVideoComponents`, `CreateVideo`, `SaveVideo` (`nodes_video.py`);
  `LTXVSeparateAVLatent` (`nodes_lt.py`), `BlockSparseAttention` (`nodes_sparse_attention.py`).

В `regen_files` остаются четыре файла, **20 853 045 168 байт**: refiner, Sharp curves, 3D upscaler,
noncausal fast VAE. Их загрузка начинается после READY. Вне `regen_files` нужны FL2VA INT8,
Qwen text encoder, video VAE и audio VAE — это прежние boot-файлы из `files`.
Ref2VA INT8 для `curve_table` остаётся в `reference_files`: загрузка после READY, а `prepare_regen`
ждёт весь reference-пакет перед постановкой графа в очередь. Недостающих загрузок для первого задания
не обнаружено. Reference LoRA и базовая Pika LoRA refine не использует; прежние пакеты сохранены
для остальных режимов и неизменного поведения при выключенном переключателе.

### Память и проверка

Сохранены `--lowvram`, динамический offload, `keep_loaded=false`, обычный VAELoader для fallback,
`temporal_chunking=true` и прежнее окружение процесса. Сторонние install-скрипты не запускаются;
requirements проверяются в ComfyUI venv с прежними runtime constraints.
Закреплён прежний h3regen: у более позднего `0979f5723ceea842ddb0bf6b5723d7d13ec2634e`
аргумент `spatial_batch` отсутствовал в INPUT_TYPES. Fast decode передаёт обязательный fallback `vae`.
Рекомендуемый запас: 128 GB RAM, диск от 180 GB; сумма boot + reference + regen весов теперь
**85 809 632 119 байт**. Это не измерение пиков VRAM/RAM.

Тесты используют синтетические MP4 и FakeComfy, проверяют 1080p/2K, исходные аудиопакеты,
границы 5–15 с, отсутствие узлов до `/prompt`, загрузки и неизменный выключенный boot/download/modes.
Локальная проверка этой правки: 887 тестов прошли без asyncio и lifecycle; 36 вариантов графа refine
полностью совпали с исходными. Полный pytest здесь зависает при создании системного socketpair
даже для пустого asyncio.run; серверные тесты требуют повторного запуска в рабочем окружении.
Эта правка не выполняет внешние сетевые запросы, GPU-запуски, развёртывание или commit/push.
Независимое Opus-review остаётся внешнему процессу SuperAI; здесь Opus недоступен.

Необязательное поле `lab` принимает только worker `/clip/start`. Студия его не отправляет.
Решение владельца от 2026-11-10 после третьего A/B: «pika6_kitchen_sol», затем «не, давай все таки pika6_kitchen )) она лучше и красивше». Прежний t4 показался «немного такая знаешь пластиковая». Без `lab` или с `lab={}` text/frames используют Pika Flex LoRA (`pika6`), 6 шагов, фиксированные сдвиги 12/3, `lcm`, `ManualSigmas` и Kitchen INT8, без SOL. Непустой `lab` сохраняет прежние значения пропущенных ключей; refs не изменён. `turbo4_768` остаётся лабораторным профилем.

Третий A/B: 48 роликов, 768p, 5 с, фотореализм, кино, мультфильм и графика, одна RTX 5090. Время выполнения: pika6 — 145 с, pika6 + Kitchen — 71 с, pika6 + Kitchen + SOL — 51 с, t4 + Kitchen + SOL — 41 с. Характер изображения связан с LoRA turbo4_768, а не с ускорителями; семейство Pika сохраняет естественные тона, речь во всех вариантах — слово в слово.

Непустой объект `lab` сохраняет прежние значения пропущенных ключей: `attention: default`, `sparse: off`, `lora: turbo8_544` для text/frames. Например, `{"lora":"turbo8_544"}` даёт 8 шагов и сдвиги 12/3 без Kitchen/SOL. Для refs исходный профиль остаётся `ref8_768`, 8 шагов, 6/3, без Kitchen/SOL.

Пример дополнения к обычному запросу:

```json
{"lab":{"attention":"kitchen","sparse":"sol","lora":"turbo8_768","shift":{"video":6,"audio":3}}}
```

`attention`: `default` (штатный backend ComfyUI) или `kitchen` (INT8).
`sparse`: `off` или `sol`. Другие значения, массив методов и неизвестные ключи отвергаются.
SOL: `selection="sol-attn"`, `selection.tau=1.3`, `start_percent=0.2`, `end_percent=1.0`,
`min_tokens=12288`, `sink_conditioning="exact_kv_and_rows"`, `dense_blocks=""`, `extra_tokens=256`, `verbose=false`.
Граф содержит не более одного sparse-узла; SLA/VSA не включаются.

| `lora` | Режим | Шаги | Video/audio shift | Переопределения |
| --- | --- | ---: | --- | --- |
| `turbo8_544` | text/frames, значение lora в непустом lab | 8 | 12/3 | steps: 4 или 8; shift |
| `turbo8_768` | text/frames | 8 | 6/3 | shift |
| `turbo4_768` | text/frames, лабораторный профиль | 4 | 6/3 | shift |
| `pika6` | text/frames, основной профиль с Kitchen (без SOL) | 6 | 12/3 | нет |
| `ref_v01` | refs | 4 | 12/3 | shift |
| `ref8_768` | refs, новое исходное значение | 8 | 6/3 | shift |

`steps` — целое число; для фиксированного профиля можно явно передать только его штатное значение.
`shift` — объект ровно с двумя конечными числами `video` и `audio` от 0.01 до 100; boolean не допускается.
Turbo-профили разрешают экспериментальные сдвиги в диапазоне узла, это не обещание качества.
Pika запрещает переопределение shift даже штатными значениями: его расписание фиксировано.
Названия 544/768 обозначают обучение LoRA; размер изображения по-прежнему выбирается полем `size`.

Для Pika используется одна Pika FL2VA LoRA без Turbo, `lcm`, `BasicGuider` (CFG 1) и `ManualSigmas`:
`1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0`.
Для остальных профилей — `res_multistep` и `BasicScheduler(simple)`.
Цепочка модели: база → LoRA → сдвиги → Kitchen, если выбран → SOL, если выбран → guider; для Turbo модель также поступает в scheduler, у Pika сигмы фиксированы.
Каждый запрос строит отдельный граф. Без `lab` и с `lab={}` у text/frames строится одинаковый основной граф со сдвигами 12/3 и Kitchen, без SOL.

Перед отправкой графа, включая основной профиль, worker читает `/object_info`. Отсутствующий узел даёт
`lab requires ComfyUI node …`; отсутствие `comfy kitchen attention` в вариантах узла —
`lab kitchen requires the comfy kitchen INT8 attention backend`. Тихий откат Kitchen на PyTorch не принимается.
Проверка происходит до `/prompt`, ошибка попадает в `/clip/state`, следующие задания могут выполняться.

Нормализованный `lab`, включая значения по умолчанию, входит в fingerprint: повтор request с изменённой
настройкой возвращает 409; отсутствие `lab` и `lab={}` эквивалентны.
Состояние возвращает нормализованный `lab`, `exec_s` (выполнение после ожидания файлов/очереди) и, если наблюдались
WS-события начала и конца узла sampler, `sampler_s`. Это время между событиями `executing` узла 16 и следующего
узла, а не измерение CUDA-ядра. Без наблюдений поле отсутствует; ноль не подставляется.

При старте загружаются пять файлов общим размером **42 030 014 335 байт**, включая `pika6` с прежними пинами Pika-Labs/Pika-H3-Flex. `turbo4_768` перенесён в `lab_files` с прежними пинами; он и `turbo8_544` загружаются по запросу. Для t4 с ускорителями передайте `{"lab":{"lora":"turbo4_768","attention":"kitchen","sparse":"sol"}}`.

Дополнительные LoRA перечислены в `lab_files` манифеста. Они не участвуют в boot или предварительной загрузке
refs: первая задача с выбранным профилем начинает фоновую загрузку после READY и ждёт со стадией
`downloading_lab_lora`. Загрузка общая для одновременных запросов одного профиля; обычные задания проходят
в очередь независимо. Ошибка заканчивает ожидающие задания; новое задание повторяет попытку. Размер, SHA-256,
диапазоны и возобновление проверяет тот же загрузчик, что и базовые файлы.
Refs по-прежнему предварительно загружает базу и новую LoRA после READY; `ref_v01` сначала ждёт этот пакет,
затем свою LoRA. Это учитывается при сравнении холодного времени; `exec_s` не включает загрузки.

## Выбор Ref2VA и пины

Выбран ref2v **8-step v1.1 768p**: восемь вычислений — исходный вариант для проверки качества;
4-step v1.0 768p — альтернатива с меньшим числом вычислений, здесь отдельным профилем не добавлена.
Это выбор для предстоящего A/B, не результат измерения качества. Прежний 544p v0.1 сохранён для сравнения.
Карточка lightx2v на закреплённой ревизии ссылается на таблицу ModelTC; таблица описывает сдвиги 6/3 для
768p FL2VA и 12/3 для прежних 544p, но отстаёт от перечня новых Ref2VA-файлов. Для новых Ref2VA применено
требование задания 6/3; существование файлов, размеры и SHA-256 проверены по HF LFS-метаданным.
Превосходство v1.1 над 4-step по движению/звуку без рендера не установлено.

Источники:

- [lightx2v, закреплённая карточка](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/30f485ed874aa44c8183348ad361c5eabad318da/README.md)
- [ModelTC, характеристики](https://github.com/ModelTC/Minimax-H3-Turbo#model-specs)
- [Pika, Comfy-формат и расписание](https://huggingface.co/Pika-Labs/Pika-H3-Flex/blob/ca45d99f9b1d1859ffb9dc1cd6bfaaa994257f56/comfy/README.md)
- [Манифест с полными путями и SHA-256](model-manifest.json)

Репозитории и commit:

- Comfy-Org/MiniMax-H3: `54472221d27c695fca9621b0c3ed52f06c4af707`.
- lightx2v/Minimax-h3-Turbo: `30f485ed874aa44c8183348ad361c5eabad318da`.
- Pika-Labs/Pika-H3-Flex: `ca45d99f9b1d1859ffb9dc1cd6bfaaa994257f56`.

| Профиль / файл | Байты | SHA-256 |
| --- | ---: | --- |
| Ref2VA INT8, Comfy-Org | 20970379616 | `9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779` |
| turbo8_544, Comfy-Org | 1956193000 | `2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e` |
| ref8_768, lightx2v | 1956193000 | `e56863a7524e0300729cbb7b64cc8b548d90698c1abbf7d83898b0122fbe1e81` |
| ref_v01, Comfy-Org | 1956193000 | `5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c` |
| turbo8_768, lightx2v | 1956193000 | `fb4c00361c94bad47f878d5e096c6cab2f45063b0968bd5756dcb7da6936f53e` |
| turbo4_768, lightx2v | 1956192992 | `c396a9a06f58399e9df9754b18299818d84a2ddd371724ba48fe4a41221437dc` |
| pika6, Pika-Labs | 1956172176 | `68783e6447696ec365d13c0554941e070bee4cfea13c37da52498ec4f9a53f89` |

## Проверка узлов по исходникам 0.44

Корень: `D:/SuperAI-data/local/ComfyUI/ComfyUI_windows_portable/ComfyUI`.
`comfyui_version.py:3` содержит `0.44.0`. Исходники прочитаны без импортов и запуска ComfyUI.
API ID сверены отдельно от отображаемых имён. Все необходимые узлы присутствуют; из-за отсутствия узлов
ни одна опция задания не исключена. GPU/backend на 5090 проверяются при будущем запуске.

| Узел | Файл:строка | Проверенные входы |
| --- | --- | --- |
| UNETLoader | nodes.py:992 | unet_name, weight_dtype=default |
| CLIPLoader | nodes.py:1017 | clip_name, type=minimax, device=default |
| VAELoader | nodes.py:776 | vae_name |
| LoadImage | nodes.py:1747 | image |
| LoraLoaderModelOnly | nodes.py:761 | model, lora_name, strength_model=1 |
| VAEDecode | nodes.py:321 | samples, vae |
| MiniMaxH3ImageToVideo | comfy_extras/nodes_minimax_h3.py:127 | clip, vae, prompt, width, height, length, first_frame, last_frame |
| MiniMaxH3ReferenceToVideo | comfy_extras/nodes_minimax_h3.py:254 | clip, vae, audio_vae, prompt, width, height, length, ref_image_size=match, ref_images.ref_image_N |
| MiniMaxH3SigmaShift | comfy_extras/nodes_minimax_h3.py:378 | model, shift_video, shift_audio; отображаемое имя ModelSamplingMiniMaxH3 |
| ModelAttentionBackend | comfy_extras/nodes_model_advanced.py:401 | model, attention="comfy kitchen attention"; доступность backend проверяется при построении schema |
| BlockSparseAttention | comfy_extras/nodes_sparse_attention.py:353 | model, selection/selection.tau, start_percent, end_percent, dense_blocks, min_tokens, extra_tokens, sink_conditioning, verbose |
| BasicScheduler | comfy_extras/nodes_custom_sampler.py:17 | model, scheduler, steps, denoise |
| KSamplerSelect | comfy_extras/nodes_custom_sampler.py:379 | sampler_name |
| BasicGuider | comfy_extras/nodes_custom_sampler.py:801 | model, conditioning; CFG=1 в comfy/samplers.py:1193 |
| RandomNoise | comfy_extras/nodes_custom_sampler.py:1003 | noise_seed |
| SamplerCustomAdvanced | comfy_extras/nodes_custom_sampler.py:1020 | noise, guider, sampler, sigmas, latent_image |
| ManualSigmas | comfy_extras/nodes_custom_sampler.py:1125 | sigmas, строка чисел |
| VAEDecodeAudio | comfy_extras/nodes_audio.py:116 | samples, vae |
| CreateVideo | comfy_extras/nodes_video.py:209 | images, fps, audio |
| SaveVideo | comfy_extras/nodes_video.py:137 | video, filename_prefix, format, format.codec, format.codec.encoding |

`comfy/samplers.py:973` включает `lcm`; `comfy_api/latest/_io.py:1231` и `:1266` описывают DynamicCombo:
в wire-графе передаются `selection` и `selection.tau`, а узел получает собранный объект selection.
`nodes_model_advanced.py:436` содержит тихий fallback отсутствующего backend; worker проверяет доступность заранее.
