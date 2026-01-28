@echo off
chcp 65001 >nul
title GS Server - Gaussian Splatting Training

echo ========================================
echo   GS Server - Gaussian Splatting
echo ========================================
echo.

cd /d "%~dp0"

REM Проверить Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python не найден! Установите Python 3.10+
    pause
    exit /b 1
)

REM Активировать venv если есть
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [OK] Virtual environment активирован
) else (
    echo [WARN] Virtual environment не найден, используется системный Python
)

REM Проверить зависимости
python -c "import fastapi" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Установка зависимостей...
    pip install -r gs_server\requirements.txt
)

REM Получить IP адрес
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    set IP=%%a
    goto :found_ip
)
:found_ip
set IP=%IP:~1%

echo.
echo [INFO] Запуск сервера...
echo.
echo   Local:   http://localhost:8080
echo   Network: http://%IP%:8080
echo   Docs:    http://localhost:8080/docs
echo.
echo   Для остановки нажмите Ctrl+C или запустите stop_server.bat
echo ========================================
echo.

REM Сохранить PID в файл
python -m gs_server

pause
