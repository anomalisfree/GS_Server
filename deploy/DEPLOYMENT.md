# Развёртывание GS Training Server на Google Cloud
### Полное руководство для начинающих

---

## Что мы делаем и зачем

**GS Training Server** — это сервер, который принимает фотографии через интернет, обрабатывает их программой COLMAP (восстанавливает 3D-геометрию), а затем обучает модель Gaussian Splatting с помощью Brush. Результат — файл `.ply`, который можно открыть в 3D-вьювере.

Мы разворачиваем его на **Google Cloud** (арендуем виртуальный компьютер с мощной видеокартой NVIDIA), потому что:
- Обучение Gaussian Splatting требует GPU (обычный ноутбук не справится за разумное время)
- Сервер в облаке доступен 24/7 из любого места

```
Ваш компьютер/телефон
        │  загружаете фото через браузер или API
        ▼
  Google Cloud VM (сервер)
        ├─ COLMAP  → восстанавливает положения камер
        ├─ Brush   → обучает 3D Gaussian Splatting
        └─ результат: model.ply (скачиваете обратно)
```

**Стоимость**: ~$0.90/час пока VM работает. Когда не нужна — выключаете, платите ~$0.02/час только за хранение диска.

---

## Что вам потребуется

1. Аккаунт Google (обычная почта Gmail подходит)
2. Банковская карта (Google требует её для верификации, даёт $300 бесплатного кредита новым пользователям)
3. Компьютер с Windows и командной строкой (PowerShell)
4. Этот проект (уже есть на вашем компьютере)

---

## Шаг 1 — Создайте аккаунт Google Cloud

1. Откройте браузер и перейдите на **https://cloud.google.com**
2. Нажмите кнопку **"Get started for free"** (или "Начать бесплатно")
3. Войдите в свой аккаунт Google
4. Заполните форму — имя, страна, тип аккаунта (выберите "Individual")
5. Введите данные карты (деньги не снимут — это только верификация)
6. Вы получите **$300 бесплатного кредита** на 90 дней — этого хватит на многие часы работы

После регистрации вы попадёте в **Google Cloud Console** (веб-интерфейс).

---

## Шаг 2 — Создайте проект в Google Cloud

Проект — это контейнер для всех ваших ресурсов (VM, диски, настройки).

1. В верхней части страницы нажмите на выпадающий список проектов (рядом с логотипом Google Cloud)
2. Нажмите **"New Project"** (Новый проект)
3. Введите имя, например: `gs-training`
4. Нажмите **"Create"**
5. Запомните или скопируйте **Project ID** — он понадобится (выглядит как `gs-training-123456`)

---

## Шаг 3 — Установите Google Cloud CLI (gcloud)

`gcloud` — это программа для управления Google Cloud из командной строки Windows.

### Установка на Windows

1. Откройте **PowerShell** (нажмите Win+X → Windows PowerShell)
2. Выполните команду:

```powershell
winget install Google.CloudSDK
```

Если `winget` не найден — скачайте установщик вручную:
- Перейдите на https://cloud.google.com/sdk/docs/install-sdk
- Скачайте **"Windows x86_64"**
- Запустите установщик, следуйте инструкциям
- **Отметьте галочку** "Run gcloud init" в конце установки

3. **Закройте и снова откройте PowerShell** (чтобы gcloud появился в PATH)

4. Проверьте установку:

```powershell
gcloud version
```

Должно вывести что-то вроде:
```
Google Cloud SDK 500.0.0
```

---

## Шаг 4 — Авторизуйтесь в Google Cloud

```powershell
gcloud auth login
```

Откроется браузер → войдите в свой аккаунт Google → разрешите доступ → вернитесь в терминал.

Теперь укажите ваш проект (замените `gs-training-123456` на ваш реальный Project ID):

```powershell
gcloud config set project gs-training-123456
```

Проверьте, что всё правильно:

```powershell
gcloud config list
```

Должно показать ваш проект в строке `project = ...`

---

## Шаг 5 — Включите биллинг для проекта

Без этого нельзя создать VM с GPU.

1. Перейдите на https://console.cloud.google.com/billing
2. Нажмите **"Link a billing account"**
3. Выберите ваш биллинг-аккаунт (создан при регистрации)
4. Нажмите **"Set account"**

---

## Шаг 6 — Создайте виртуальную машину (VM) с GPU

Откройте PowerShell, перейдите в папку с проектом:

```powershell
cd D:\VSProjects\GS_Sever
```

Запустите скрипт создания VM:

```powershell
bash deploy/create_vm.sh
```

> **Если bash не работает в PowerShell** — установите Git for Windows (https://git-scm.com), он включает bash. Затем откройте "Git Bash" и выполните те же команды.

Скрипт создаст:
- Виртуальный компьютер с **8 процессорными ядрами** и **32 ГБ RAM**
- Видеокарту **NVIDIA L4** (24 ГБ видеопамяти)
- Системный диск **100 ГБ** (Ubuntu Linux)
- Отдельный диск **500 ГБ** для ваших данных (задания, модели)
- Правило файрвола — открывает порт 8080 для доступа к серверу

Создание занимает **1–3 минуты**. В конце увидите внешний IP-адрес вашей VM.

### Если появится ошибка "GPU quota"

Google Cloud по умолчанию ограничивает новые аккаунты (quota = 0 GPU). Нужно запросить увеличение:

1. Перейдите на https://console.cloud.google.com/iam-admin/quotas
2. В поиске введите `NVIDIA_L4_GPUS`
3. Выберите регион `us-central1`
4. Нажмите **"Edit Quotas"** → запросите лимит `1`
5. Заполните форму (опишите зачем вам GPU, например "ML training")
6. Ждите письма на почту (обычно одобряют за несколько часов)

---

## Шаг 7 — Загрузите проект на сервер

Следующая команда копирует все файлы проекта с вашего компьютера на VM:

```bash
gcloud compute scp --recurse . gs-training-server:~/gs_server \
  --zone=us-central1-a \
  --exclude=".git,target,.venv,jobs,uploads,models,data,colmap"
```

**Что значит эта команда:**
- `scp` — копирование файлов по SSH
- `--recurse` — копировать папки рекурсивно
- `.` — текущая папка (проект на вашем компьютере)
- `gs-training-server:~/gs_server` — папка `gs_server` на VM
- `--exclude=...` — не копировать скомпилированные файлы (они большие и не нужны)

Загрузка займёт **1–5 минут** в зависимости от скорости интернета.

---

## Шаг 8 — Подключитесь к серверу по SSH

SSH — это защищённое подключение к командной строке удалённого компьютера.

```bash
gcloud compute ssh gs-training-server --zone=us-central1-a
```

При первом подключении спросит создать SSH-ключ — нажмите Enter (оставьте пустой пароль).

Вы увидите приглашение командной строки Linux, например:
```
yourname@gs-training-server:~$
```

Теперь вы "внутри" вашей VM в Google Cloud. Все следующие команды выполняются **на VM**.

---

## Шаг 9 — Установите все компоненты

Запустите установочный скрипт:

```bash
bash ~/gs_server/deploy/install.sh
```

Скрипт автоматически:

| Шаг | Что происходит | Время |
|-----|---------------|-------|
| 1 | Обновляет список пакетов Ubuntu | ~1 мин |
| 2 | Устанавливает NVIDIA драйверы и CUDA | ~5 мин |
| 3 | Устанавливает Vulkan, COLMAP, Python | ~3 мин |
| 4 | Устанавливает Rust (язык программирования) | ~2 мин |
| 5 | **Компилирует Brush** из исходного кода | **10–20 мин** |
| 6 | Устанавливает Python-зависимости сервера | ~2 мин |
| 7 | Подключает диск данных к `/data` | ~1 мин |
| 8 | Создаёт системный сервис (автозапуск) | ~1 мин |

> **Важно — перезагрузка после NVIDIA драйверов:**
>
> Если скрипт выведет сообщение:
> ```
> NVIDIA drivers installed. Reboot required.
> Run: sudo reboot
> ```
> Выполните `sudo reboot`, подождите 2 минуты, снова подключитесь (`gcloud compute ssh ...`) и запустите скрипт повторно — он продолжит с того места, где остановился.

По завершении увидите:
```
============================================================
  GS Training Server is running!
============================================================
  API:  http://12.34.56.78:8080
  Docs: http://12.34.56.78:8080/docs
============================================================
```

---

## Шаг 10 — Проверьте работу сервера

### На VM (в SSH-сессии):

```bash
# Проверить статус сервиса
sudo systemctl status gs-server

# Проверить ответ сервера
curl http://localhost:8080/status
```

Ожидаемый ответ:
```json
{"status": "ready", "version": "...", "active_jobs": 0}
```

### С вашего компьютера:

Узнайте внешний IP вашей VM:

```bash
gcloud compute instances describe gs-training-server \
  --zone=us-central1-a \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

Откройте в браузере: **`http://ВНЕШНИЙ_IP:8080/docs`**

Вы увидите интерактивную документацию API (Swagger UI), где можно прямо в браузере:
- Загружать фото
- Запускать обучение
- Следить за прогрессом
- Скачивать готовую модель

---

## Как пользоваться сервером

### Вариант 1 — Через браузер (Swagger UI)

Откройте `http://ВНЕШНИЙ_IP:8080/docs` и используйте кнопки прямо на странице.

### Вариант 2 — Через Python (клиентский пример)

```python
import requests

SERVER = "http://ВНЕШНИЙ_IP:8080"

# 1. Создать задание
resp = requests.post(f"{SERVER}/jobs", json={"name": "my_scene"})
job_id = resp.json()["job_id"]

# 2. Загрузить фотографии
with open("photo1.jpg", "rb") as f:
    requests.post(f"{SERVER}/jobs/{job_id}/upload", files={"file": f})

# 3. Запустить обучение
requests.post(f"{SERVER}/jobs/{job_id}/start")

# 4. Проверить прогресс
import time
while True:
    status = requests.get(f"{SERVER}/jobs/{job_id}").json()
    print(status["progress"]["message"])
    if status["status"] in ("completed", "failed"):
        break
    time.sleep(10)

# 5. Скачать модель
with open("model.ply", "wb") as f:
    f.write(requests.get(f"{SERVER}/jobs/{job_id}/download").content)
```

---

## Управление расходами

GPU-VM стоит ~$0.90/час. Когда она вам не нужна — **обязательно останавливайте её**.

| Действие | Команда | Стоимость |
|----------|---------|-----------|
| Остановить VM | `gcloud compute instances stop gs-training-server --zone=us-central1-a` | ~$0.02/час (только диск) |
| Запустить VM | `gcloud compute instances start gs-training-server --zone=us-central1-a` | ~$0.90/час |
| Удалить VM (диск остаётся) | `gcloud compute instances delete gs-training-server --zone=us-central1-a --keep-disks=data` | $0 |
| Удалить диск с данными | `gcloud compute disks delete gs-data-disk --zone=us-central1-a` | $0 |

> **Совет**: настройте бюджетные уведомления в Google Cloud Console (Billing → Budgets & Alerts), чтобы получать письмо при достижении определённой суммы.

После остановки VM ваш сервер недоступен. После запуска — нужно подождать ~30 секунд пока сервис поднимется (он стартует автоматически).

---

## Обновление кода сервера

Если вы изменили файлы в `gs_server/` на своём компьютере:

```bash
# С вашего компьютера — загрузить обновлённый код
gcloud compute scp --recurse ./gs_server gs-training-server:~/gs_server/ \
  --zone=us-central1-a

# Перезапустить сервер
gcloud compute ssh gs-training-server --zone=us-central1-a \
  --command="sudo systemctl restart gs-server"
```

---

## Полезные команды на VM

```bash
# Смотреть логи в реальном времени
sudo journalctl -fu gs-server

# Перезапустить сервер
sudo systemctl restart gs-server

# Проверить использование GPU
nvidia-smi

# Посмотреть список заданий
ls /data/jobs/

# Проверить место на диске
df -h /data
```

---

## Если что-то пошло не так

### Сервер не отвечает

```bash
# На VM проверьте статус
sudo systemctl status gs-server

# Посмотрите последние ошибки
sudo journalctl -n 50 -u gs-server
```

### Ошибка при компиляции Brush

```bash
# Очистить и пересобрать
cd ~/gs_server/brush
cargo clean
cargo build --release -p brush-app
sudo systemctl restart gs-server
```

### Порт 8080 не доступен снаружи

```bash
# С вашего компьютера проверьте правила файрвола
gcloud compute firewall-rules list --project=ВАШ_PROJECT_ID
```

Должно быть правило `allow-gs-server` с `tcp:8080`. Если нет:

```bash
gcloud compute firewall-rules create allow-gs-server \
  --direction=INGRESS --action=ALLOW \
  --rules=tcp:8080 --target-tags=gs-server
```

---

## Переменные окружения (настройки сервера)

Все настройки хранятся в файле `/etc/systemd/system/gs-server.service` на VM.

Чтобы изменить, например, количество шагов обучения:

```bash
sudo nano /etc/systemd/system/gs-server.service
```

Найдите строку `Environment="BRUSH_TOTAL_STEPS=30000"` и измените значение. Затем:

```bash
sudo systemctl daemon-reload
sudo systemctl restart gs-server
```

| Переменная | Значение по умолчанию | Описание |
|------------|----------------------|----------|
| `GS_PORT` | `8080` | Порт сервера |
| `GS_COLMAP_EXE` | `/usr/bin/colmap` | Путь к COLMAP |
| `GS_BRUSH_EXE` | `~/gs_server/brush/target/release/brush` | Путь к Brush |
| `COLMAP_USE_GPU` | `true` | Использовать GPU для COLMAP |
| `MASKING_ENABLED` | `true` | Удалять людей/машины с фото перед обучением |
| `BRUSH_TOTAL_STEPS` | `30000` | Количество итераций обучения |
| `BRUSH_MAX_RESOLUTION` | `1920` | Максимальное разрешение изображений |

---

## Безопасность

Сейчас сервер открыт для всего интернета — **любой знающий IP-адрес может им воспользоваться**. Для личного использования это нормально, но для продакшена ограничьте доступ только вашим IP:

```bash
# Узнайте ваш внешний IP: https://whatismyip.com
MY_IP="123.45.67.89"

# Удалите старое правило
gcloud compute firewall-rules delete allow-gs-server

# Создайте новое — только для вашего IP
gcloud compute firewall-rules create allow-gs-server \
  --direction=INGRESS --action=ALLOW \
  --rules=tcp:8080 --target-tags=gs-server \
  --source-ranges="${MY_IP}/32"
```
