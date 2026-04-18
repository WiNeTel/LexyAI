@echo off
title Lexy AI - WhatsApp Bridge (Baileys)
echo ============================================
echo   Lexy AI - WhatsApp Bridge
echo   Phone: +49 176 211 05176
echo   Port:  3000
echo ============================================
echo.

cd /d "%~dp0..\bridges\whatsapp"

:: Install dependencies on first run
if not exist "node_modules" (
    echo [bridge] Installing dependencies...
    call npm install
    echo.
)

:: Start the bridge
echo [bridge] Starting Baileys bridge...
echo [bridge] If this is the first start, a QR code will appear.
echo [bridge] Scan it with Lexy's WhatsApp phone.
echo.
node bridge.js --verbose

pause
