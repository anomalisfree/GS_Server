# Как продолжить работу над проектом на другом компьютере

Состояние на **2026-10-19**. Документ для владельца проекта; технические подробности для Claude Code лежат в
`CLAUDE.md` (его Claude читает сам при открытии папки проекта).

## 0. Коротко (чек-лист)

1. `git clone https://github.com/anomalisfree/GS_Server.git` (код, документация, `tools/`, `patches/`).
   **Важно:** на момент передачи последние 11 коммитов (включая эту инструкцию) **не отправлены на GitHub**:
   `git push` с этого ПК упал с «Invalid username or token». Сначала выполните вход (раздел 2.1) и
   `git push origin main`. Если GitHub недоступен, возьмите готовый файл
   `C:\Space\Projects\VSProjects\GS_Sever_backup\GS_Server_full_2026-10-19.bundle` (45 МБ, вся история),
   скопируйте на новый ПК и выполните `git clone GS_Server_full_2026-10-19.bundle GS_Server`.
2. Установить Python 3.11 и Google Cloud SDK; `pip install -r tools/requirements.txt`.
3. `gcloud auth login alex.lisavets@q3dsensing.com`, `gcloud config set project dgs-training`.
4. Скопировать вручную то, чего нет в git: папку `data/` (минимум `data/CameraSet`, 1,3 ГБ + `results` 0,2 ГБ).
5. Узнать адрес сервера и задать `GS_SERVER` (раздел 3), проверить `curl http://<IP>:8080/status`.
6. Открыть папку в Claude Code и начать с чтения `CLAUDE.md`, затем `CameraSet_Training_Report_2026-10-19.md`.

## 1. Что где лежит

| Что | Где | В git? |
|---|---|---|
| Код сервера `gs_server/`, скрипты конвертации (`leica_to_nerfstudio.py` и др.) | репозиторий | да |
| Инструкции: `COMPLETE_USER_GUIDE_EN.html/.md` (приложение + сервер), `SERVER_USER_MANUAL*.md/.html`, `CLIENT_*_GUIDE.md`, `README.md` | репозиторий | да |
| Отчёты (`AU_Las_Vegas_*`, `Batch12_*`, `Core_Artifact_*`, `CameraSet_Training_Report_2026-10-19.md`) | репозиторий | да |
| Инструменты для работы с сервером: `tools/` (нарезка видео, создание задач, ожидание, диагностика, маски неба, сборка документации) | репозиторий | да |
| **Rust-правки Brush** (глубина, anchor, aniso, photo-consistency): `patches/` (patch + bundle) | репозиторий | да, но **живая копия на сервере** (раздел 5) |
| Каталог `brush/` (чужой репозиторий upstream) | локально, в `.gitignore` | нет |
| Исходные данные и результаты: `data/` (4,8 ГБ: `CameraSet`, `data_prepared`, сканы...) | локально, в `.gitignore` | **нет, копировать вручную** |
| Рабочие каталоги сервера `jobs/ uploads/ models/`, бинарники COLMAP `colmap/bin`, `.venv` | локально | нет |
| Модели, видео, 105 завершённых задач | на сервере (`/data/jobs`) | нет, скачиваются через API |
| Память и история сеансов Claude Code | `C:\Users\anoma\.claude\projects\...` на старом ПК | нет (раздел 6) |

## 2. Установка на новом компьютере

Два варианта работы. **Обычно хватает первого**: тяжёлые вычисления идут на сервере в Google Cloud.

### Вариант А: только клиент (рекомендуется)

```powershell
winget install Python.Python.3.11
winget install Google.CloudSDK
git clone https://github.com/anomalisfree/GS_Server.git
cd GS_Server
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r tools/requirements.txt
```

Нужны: `requests` (отправка задач), `opencv-python-headless` + `numpy` (нарезка видео), `markdown` (сборка HTML-документов).

### Вариант Б: запускать сервер локально (нужен NVIDIA GPU)

Дополнительно к варианту А: `pip install -r gs_server/requirements.txt` (в нём torch), установить COLMAP
(Windows: распаковать в `colmap/`, Linux: `apt install colmap`) и собрать Brush (раздел 5, нужен Rust).
Подробности и переменные окружения: `README.md`. Для обычной работы это не нужно.

### 2.1 Вход в GitHub (чтобы работали `git push` и `git pull`)

Парольный вход GitHub не поддерживает. Любой из способов:

* `winget install GitHub.cli`, затем `gh auth login` (браузер) и `gh auth setup-git`;
* или Personal Access Token (GitHub -> Settings -> Developer settings), его вводят вместо пароля при первом
  `git push`; Git Credential Manager (идёт с Git for Windows) запомнит его.

Токены и пароли никому (в том числе Claude) не передавайте в чат: вход выполняйте сами. Пока вход не
сделан, `git push origin main` с этого ПК отправляет в Git только ошибку авторизации, ничего не теряется.

## 3. Доступ к серверу (Google Cloud)

* VM `gs-training-server`, зона `us-central1-a`, проект `dgs-training`, сервис `gs-server` на порту 8080.
* Вход (откроется браузер, нужны вы лично): `gcloud auth login alex.lisavets@q3dsensing.com`.
  Аккаунт `anomalisfree@gmail.com` прав на VM не имеет. **Токен gcloud истекает за считанные часы** (бывает
  между двумя запросами подряд), при ошибке «Reauthentication failed» просто войдите заново. Запускайте только
  один вход за раз.
* Внешний IP (он может меняться при остановке VM):

  ```powershell
  gcloud compute instances describe gs-training-server --zone=us-central1-a --project=dgs-training `
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
  $env:GS_SERVER = "http://<IP>:8080"
  curl.exe $env:GS_SERVER/status
  ```

* Команды на VM: `gcloud compute ssh gs-training-server --zone=us-central1-a --project=dgs-training --quiet --command="..."`.
  Сложные команды **не пишите в одну строку** (PowerShell портит кавычки и `$`): положите скрипт в файл,
  `gcloud compute scp`, затем запустите. Строка `--command` в Git-Bash не должна начинаться с `/`.
* Порт 8080 открыт всем без пароля (только firewall): не публикуйте ссылки на сервер.

Типичные операции (подробности и ловушки в `CLAUDE.md`):

```bash
python tools/client/video_to_frames.py --out frames8 --block 8 --video "<файл>.MOV=scan01"
python tools/client/submit_job.py --name <имя> --mode colmap_images --frames frames8 --colmap-config '{"single_camera": true}'
python tools/client/wait_for_job.py <JOB_ID> --preview --out results --label <имя>
```

Перед `systemctl restart gs-server` **всегда** проверьте, нет ли активной задачи
(`curl $GS_SERVER/status` -> `active_jobs`), иначе перезапуск убьёт её без возможности продолжить.

## 4. Данные, которых нет в git

Скопируйте на новый компьютер (флешка, облако, `robocopy`): папку `GS_Sever\data\` (4,8 ГБ), как минимум:

* `data\CameraSet\` (1,3 ГБ): 2 видео `.MOV`, 6 `.ply`, 6 `.dp` (исходные данные эксперимента);
* `data\CameraSet\results\` (0,2 ГБ): модели `.ply` и видео-пролёты пяти обученных задач, `jobs_cleanup_candidates.csv`;
* остальные сканы и `data_prepared\` при необходимости.

Если `results\` потерян, модели и видео всё равно можно скачать, пока задачи есть на сервере (идентификаторы в
`CameraSet_Training_Report_2026-10-19.md`): `GET /jobs/<id>/model`, `GET /jobs/<id>/preview/download`.

## 5. Brush (Rust-правки): единственный «хрупкий» элемент

Правки Brush **не в основном репозитории**. Живая копия: на сервере `/home/anoma/brush`, ветка
`gs-server-patches`, коммит `b0900e6` поверх upstream `3edecbb` (релиз v0.3.0). Копия-страховка лежит в
`patches/` (патч проверен: применяется чисто к `3edecbb`). Если диск VM потерян, правки восстанавливаются так:

```bash
git clone https://github.com/ArthurBrussee/brush.git brush-work
cd brush-work && git checkout 3edecbb
git apply ../patches/brush_gs-server-patches.patch      # или: git am ../patches/...patch
cargo build --release --bin brush_app                   # на сервере: source ~/.cargo/env первым
```

Локальный каталог `brush/` в проекте стоит на другом коммите (`83b7c71f`) и содержит 37 строк старых
правок в `config.rs`/`train.rs`; патч на него **не применять**, делайте отдельный клон, как выше.
Менять Rust нужно только для новых опций обучения. После любой правки на сервере обновите копию в `patches/`
(команды в `patches/README.md`) и закоммитьте.

## 6. Claude Code на новом компьютере

* Откройте папку проекта: `CLAUDE.md` подхватится автоматически, в нём всё нужное для эксплуатации сервера.
* Память и история сеансов лежат локально (`C:\Users\anoma\.claude\projects\C--Space-Projects-VSProjects-GS-Sever\`)
  и на новый компьютер сами не переезжают. Всё важное продублировано в `CLAUDE.md` и отчётах, так что
  переносить её не обязательно. Если хочется сохранить историю, скопируйте эту папку в аналогичную на новом ПК
  (имя папки строится из пути проекта, при другом пути оно будет другим).

## 7. Состояние на момент передачи

* Сервер: 106 задач (105 завершены, 1 `training_failed`), активных нет, диск `/data` заполнен примерно на 88 %
  (на 2026-10-15 свободно ~60 ГБ; проверьте `df -h /data` перед большими задачами).
* **Не удалены 27 задач (102 ГБ)**, на которые ссылаются отчёты и `config.py` (список в разделе 4.1 отчёта
  CameraSet). Решение «удалять или нет» за вами.
* Больше всего места (257 ГБ) занимают промежуточные `model_<шаг>.ply`; их удаление без потери задач
  требует работы на сервере (см. отчёт, раздел 4.1).
* Остатки моих вспомогательных файлов на VM: `/data/datasets/CameraSet_sky8` (нужен задаче `fc7ec144` для превью),
  `/home/anoma/cs_*.py|.sh`, `/home/anoma/cs_sky/`, `/tmp/cs_comp_*`.
* Результаты CameraSet и выводы: `CameraSet_Training_Report_2026-10-19.md`. Идеи на будущее: раздел 6 отчёта.
* Репозиторий: ветка `main`, remote `origin` = GitHub `anomalisfree/GS_Server`.

## 8. Проверка, что всё работает

```bash
curl $GS_SERVER/status                                   # {"status":"running",...}
python tools/client/wait_for_job.py fc7ec144-f40d-4459-8d92-34390e0b238f   # покажет "completed"
python tools/client/submit_job.py --name t --mode colmap_images --frames manual_images --dry-run
python tools/docs/build_guide_html.py                    # пересоберёт COMPLETE_USER_GUIDE_EN.html (без ошибок ссылок)
```

## 9. Куда читать дальше

| Нужно | Файл |
|---|---|
| Как работать с сервером, ловушки, деплой | `CLAUDE.md` |
| Полное описание приложения и сервера, API, параметры | `COMPLETE_USER_GUIDE_EN.html` |
| Что известно про качество моделей | `AU_Las_Vegas_Floating_Tile_Artifact_Report_2026-09-24.md`, `CameraSet_Training_Report_2026-10-19.md` |
| Инструменты | `tools/README.md` |
| Правки Brush | `patches/README.md` |
