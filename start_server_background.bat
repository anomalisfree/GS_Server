@echo off
chcp 65001 >nul
title GS Server - Background Mode

echo ========================================
echo   GS Server - Фоновый режим
echo ========================================
echo.

cd /d "%~dp0"

REM Активировать venv
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Получить IP
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    set IP=%%a
    goto :found
)
:found
set IP=%IP:~1%

echo [INFO] Запуск сервера в фоновом режиме...
echo.
echo   Local:   http://localhost:8080
echo   Network: http://%IP%:8080
echo   Docs:    http://localhost:8080/docs
echo.
echo   Для остановки запустите stop_server.bat
echo.

REM Запуск в фоне через start
start /B /MIN "GS_Server" python -m gs_server > logs\server.log 2>&1

REM Создать папку для логов
if not exist "logs" mkdir logs

echo [OK] Сервер запущен в фоновом режиме
echo [INFO] Логи: logs\server.log
echo.

timeout /t 3 >nul
