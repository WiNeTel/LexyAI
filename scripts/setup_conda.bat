@echo off
:: Lexy AI – Conda Environment Setup
echo === Lexy AI – Conda Environment Setup ===
echo.

where conda >nul 2>&1
if errorlevel 1 (
    echo [FEHLER] Conda nicht gefunden. Bitte zuerst Miniconda oder Anaconda installieren.
    exit /b 1
)

call conda env list | findstr /C:"lexyai " >nul
if errorlevel 1 (
    echo Erstelle Conda-Environment "lexyai"...
    call conda create -n lexyai python=3.11 -y
) else (
    echo [OK] Environment "lexyai" existiert bereits.
)

echo.
echo Installiere Dependencies aus requirements.txt...
call conda activate lexyai
call pip install -r requirements.txt

echo.
echo === Setup fertig. Start mit: scripts\start_lexy.bat ===
pause
