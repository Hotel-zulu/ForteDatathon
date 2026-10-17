# Контроль кассовой дисциплины по видео — ForteBank Datathon 26

AI-сервис: загружаете видео кассы → вердикт, тип нарушения, таймкод, кадр-доказательство и норматив РК.
Полностью локальный (без облачных AI API), доступен через веб-интерфейс и API для жюри.

## Быстрый старт

```bash
bash setup.sh                     # macOS/Linux (Windows: setup.ps1). Нужны Python 3.10/3.11 и ffmpeg
source .venv/bin/activate
python -m src.labels              # разметка open_labels.xlsx -> configs/labels.csv (проверка имён/таймкодов)
make serve                        # http://localhost:8000  (UI)  и  /api/v1/...
make tunnel                       # публичный URL через cloudflared (для жюри)
```

## Работа в команде

Репозиторий приватный. **Видео, разметка организаторов, документ кейса, кадры и веса моделей в git не хранятся**
(данные можно использовать только в рамках хакатона). Их передаём напрямую: AirDrop, флешка или общий диск команды.

```bash
git clone https://github.com/Hotel-zulu/ForteDatathon.git && cd ForteDatathon
# положить данные организаторов: open/videos/*.mov и open/open_labels.xlsx
bash setup.sh && source .venv/bin/activate
pip install -r requirements-ml.txt          # torch + ultralytics (долго при медленном интернете)
curl -L -o models/yolo11s.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
```

Правила:
- `git pull` перед началом работы; маленькие коммиты; не коммитить данные (`.gitignore` это страхует).
- Большие изменения — в отдельной ветке → Pull Request → `main`.
- Разметка ведётся в Label Studio на одном компьютере (см. `annotation/README.md`).

## Структура

```
open/                    данные организаторов (НЕ в git): videos/*.mov, open_labels.xlsx
configs/
  regulations.yaml       тип нарушения -> документ, пункт, цитата. ЕДИНСТВЕННЫЙ источник ссылок
  labels.csv             нормализованная разметка (генерируется)
src/
  pipeline.py            analyze(video) -> {verdict, violations[...]} — главный вход
  video_io.py            чтение видео, кадры по реальному времени, точный кадр по таймкоду
  detectors/             perception (модели) + rule-детекторы + verifier (VLM)
    base.py              интерфейсы FrameObs / Candidate / Detector / Verifier
    examples.py          примеры правил: уход с рабочего места, телефон
  evidence.py            кадр-доказательство: размытие лиц, рамки, JPEG base64
  rules.py               загрузка regulations.yaml, формирование описания
  labels.py              парсинг разметки (X/Х, «1минута 8 секунда»)
  schemas.py             JSON-схема ответа из ТЗ
api/main.py              POST /api/v1/analyze, GET /api/v1/jobs/{id}, X-API-Key, очередь задач
app/index.html           веб-интерфейс (раздаётся тем же сервером на /)
scripts/
  extract_frames.py      кадры для разметки (10 fps, или только вокруг нарушений)
  evaluate.py            качество на открытых видео: accuracy, пропуски, FP, ошибка таймкода
  api_smoke_test.py      проверка API «как у жюри», с другой машины
docs/                    ARCHITECTURE, ANNOTATION (что видно в роликах), CHECKLIST
models/                  веса моделей (НЕ в git), README с лицензиями
```

## Как развивать

1. **Разметить типы**: просмотреть ролики, заполнить `docs/ANNOTATION.md` (`make frames-around` помогает).
2. **Справочник**: для подтверждённых типов в `configs/regulations.yaml` поставить `enabled: true`,
   сверить цитату на adilet.zan.kz → `verified: true`.
3. **Perception**: реализовать в `src/detectors/__init__.py → build_perception()` (люди, роли, предметы, трекинг).
4. **Детекторы**: по одному классу на тип нарушения (см. `examples.py`).
5. **Verifier**: локальная VLM подтверждает кандидатов и пишет описание → меньше ложных срабатываний.
6. После каждого изменения: `make eval`. Цель — 0 пропусков, минимум FP, ошибка таймкода < 1–2 c.

## Сейчас

Каркас работает end-to-end, но perception — заглушка, а все типы в справочнике выключены,
поэтому сервис пока всегда отвечает `no_violation`. Это ожидаемо.
