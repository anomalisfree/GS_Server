@echo off
chcp 65001 >nul
title Stop GS Server

echo ========================================
echo   Остановка GS Server
echo ========================================
echo.

REM Найти и завершить процесс uvicorn/python с gs_server
echo [INFO] Поиск процессов сервера...

REM Метод 1: Через taskkill по имени окна
taskkill /FI "WINDOWTITLE eq GS Server*" /F >nul 2>&1

REM Метод 2: Найти python процессы с gs_server
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| findstr "PID:"') do (
    wmic process where "ProcessId=%%a" get CommandLine 2>nul | findstr "gs_server" >nul
    if not errorlevel 1 (
        echo [INFO] Завершение процесса PID: %%a
        taskkill /PID %%a /F >nul 2>&1
    )
)

REM Метод 3: Убить все uvicorn процессы (если ничего другое не помогло)
tasklist /FI "IMAGENAME eq python.exe" 2>nul | findstr /i "python" >nul
if %errorlevel% equ 0 (
    for /f "tokens=2" %%a in ('wmic process where "commandline like '%%uvicorn%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
        echo [INFO] Завершение uvicorn PID: %%a
        taskkill /PID %%a /F >nul 2>&1
    )
)

echo.
echo [OK] Сервер остановлен
echo.

timeout /t 2 >nul
