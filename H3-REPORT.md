# H3: исправления ревью и лаборатория ускорителей

Работа выполнена в существующем worktree без commit/push. Изменения ограничены `comfy-worker/`, `src/video/`,
`shared/mentions.ts` и `test/`; исходные незакоммиченные изменения сохранены.
ComfyUI, модели, GPU и Vast не запускались. Веса не скачивались, платных вызовов не было.
Сеть использовалась для Python-пакетов тестового окружения, карточек и метаданных файлов.

| Пункт | Результат и файлы | Проверки |
| --- | --- | --- |
| A1 | Общий `PICTURE_TAG` в `shared/mentions.ts`; используется в `src/video/video-work.ts` и `src/video/vendors/vast.ts`. Image/Picture, регистр и пробел перед номером поддерживаются одинаково. | `test/video-vast.test.ts`: преобразование всех форм на transport, номера вне диапазона на transport и quote; `test/video-round2.test.ts`: общий MiniMax использует Picture-форму. |
| A2 | На Vast замена и проверка только для references; text/frames сохраняют текст тегов. Те же два файла. | `Vast text and frames preserve literal picture tags from quote to transport…`: оба режима через desk и fake transport. |
| A3 | `src/video/vast-price.ts`, `src/video/catalog.ts`: references добавляет 460 секунд при quote. Кэш неизвестен, поэтому оценка предполагает холодный пакет. 22 926 572 616 байт / 50 000 000 байт/с ≈ 458.53 с, округлено до 460. | `references preserve picture numbering…`: точная прибавка USD; `desk references…budget the cold reference download`: typical и worst с коэффициентом 4. |
| A4 | Исправлен ложноположительный тест с повторяющимися ID в `test/video-vast.test.ts`. | Пять разных сохранённых кадров; отдельно четыре кадра и именованный герой. Оба случая проверяют точный detail `Слишком много референсов: 5 из 4`; запрос к поставщику не выполняется. |
| A5 | `promptMax: 4000` в Vast-маршруте `src/video/catalog.ts`. | Quote принимает 4000 и отказывает на 4001 с точным сообщением во всех трёх режимах, число денежных вызовов не меняется. |
| B | `comfy-worker/graph.py`, `lab.py`, `model-manifest.json`, `download.py`: Ref2VA по умолчанию — lightx2v ref2v 8-step v1.1 768p, shifts 6/3. `ref_v01` сохранён. | `tests/test_refs.py`: default graph, shifts, refs, pins, фоновая загрузка; `tests/test_lab.py`: ref_v01 и новый default во всех сочетаниях attention/sparse. |
| C: контракт | `contract.py`, `lab.py`: строгая форма lab, режимы, шаги, shifts; нормализованные поля в fingerprint. | `test_strict_lab_validation`, несовместимые LoRA, shift/steps, `test_steps_override_and_normalized_fingerprint`; HTTP повтор и 409 в тестах загрузки. |
| C: граф | `graph.py`: шесть профилей, Kitchen после LoRA, SOL без другого sparse, отдельный Pika lcm/ManualSigmas/CFG 1. | `test_profiles_and_accelerators`: 24 сочетания; переопределения и отсутствие переноса настроек в следующий граф. |
| C: доступность | `comfy_client.py`: проверка `/object_info` до `/prompt`, фиксированные сообщения при отсутствии узла или Kitchen backend. | `test_missing_optional_nodes_fail_before_submission`, `test_kitchen_must_not_silently_fall_back`, ошибка в state и восстановление следующего задания. |
| C: загрузки | `download.py`, `reference_models.py`, `server.py`: дополнительные LoRA только при первом использовании, по одной фоновой загрузке на профиль; стадия `downloading_lab_lora`. | Для всех четырёх дополнительных LoRA: отсутствие загрузки при boot, точные pinned URL и destination; ожидание READY, разделяемая загрузка, обычная задача проходит, отказ/повтор, успешный кэш. |
| C: измерения | `comfy_client.py`, `server.py`: state содержит exec_s, наблюдаемый sampler_s и нормализованный lab. | WS-тест считает только события нужного prompt; state после выполнения; отсутствие выдуманного sampler_s без наблюдений. |

Полные параметры, все размеры/SHA-256/commit и таблица **20 проверенных узлов с file:line и входами** —
в [LAB.md](LAB.md). Исходники ComfyUI 0.44 прочитаны статически; доступность GPU-backend проверяется при будущем запуске.
Опций задания, исключённых из-за отсутствующего узла, нет. Из двух кандидатов нового Ref2VA выбран 8-step;
альтернативный ref2v 4-step v1.0 768p не добавлен как третий refs-профиль.

Проверки:

- `python -m pytest comfy-worker/ -q --tb=short` в изолированном тестовом окружении: **900 passed**, без пропусков.
- `node --test test/video-vast.test.ts test/video-routes.test.ts test/video-round2.test.ts`: **45 passed**, без пропусков.
- `npm.cmd run typecheck`: успешно, оба TypeScript-проекта.
- `npm.cmd run lint`: успешно.

Для Python временный каталог направлен в игнорируемый `.tmp/` worktree: стандартный Windows sandbox Temp
не позволял pytest создавать каталоги. Тесты не ослаблены; сетевые тесты используют loopback и заглушки,
Python-fixture запрещает внешние подключения.

Осталось за пределами этого исполнения: реальный A/B на 5090 и независимое ревью Opus у conductor.
Opus в доступных инструментах исполнителя отсутствует; другое семейство не выдаётся за Opus.
Скорость, расход VRAM и сравнительное качество новых профилей здесь не измерялись.
