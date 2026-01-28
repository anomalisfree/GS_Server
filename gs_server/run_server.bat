@echo off
REM GS Server - Gaussian Splatting Training Server
REM Запуск сервера на Windows

setlocal

REM Перейти в директорию скрипта
cd /d "%~dp0"

REM Проверить наличие Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

REM Проверить/создать виртуальное окружение
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Активировать виртуальное окружение
call venv\Scripts\activate.bat

REM Установить зависимости если нужно
pip show fastapi >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Настройки по умолчанию
if not defined GS_HOST set GS_HOST=0.0.0.0
if not defined GS_PORT set GS_PORT=8080

echo.
echo ========================================
echo   GS Server - Gaussian Splatting
echo ========================================
echo   Host: %GS_HOST%
echo   Port: %GS_PORT%
echo   Docs: http://localhost:%GS_PORT%/docs
echo ========================================
echo.

REM Запустить сервер
python -m gs_server

pause
