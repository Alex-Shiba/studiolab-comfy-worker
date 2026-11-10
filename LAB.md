# Лаборатория H3 для A/B на 5090

Необязательное поле `lab` принимает только worker `/clip/start`. Студия его не отправляет.
Решение владельца от 2026-11-10 после третьего A/B: «pika6_kitchen_sol». Прежний t4 показался «немного такая знаешь пластиковая». Без `lab` или с `lab={}` text/frames используют Pika Flex LoRA (`pika6`), 6 шагов, фиксированные сдвиги 12/3, `lcm`, `ManualSigmas`, Kitchen INT8 и SOL. Непустой `lab` сохраняет прежние значения пропущенных ключей; refs не изменён. `turbo4_768` остаётся лабораторным профилем.

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
| `pika6` | text/frames, основной профиль с Kitchen/SOL | 6 | 12/3 | нет |
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
Каждый запрос строит отдельный граф. Без `lab` и с `lab={}` у text/frames строится одинаковый основной граф со сдвигами 12/3, Kitchen и SOL.

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
