@echo off
echo === Lexy AI — Obsidian Junction Setup ===
echo.
echo Erstellt eine Junction von docs\ zum Obsidian Vault.
echo Claude Code kann dann ueber docs\ die Knowledge Base lesen UND schreiben.
echo.

if exist "G:\AI\LexyAI\docs" (
    echo [OK] Junction existiert bereits: G:\AI\LexyAI\docs
    dir "G:\AI\LexyAI\docs\*.md" 2>nul
) else (
    echo Erstelle Junction...
    mklink /J "G:\AI\LexyAI\docs" "G:\DATEN\Obsidian\KI-Memory\KI-Memory\10-projects\lexy"
    if errorlevel 1 (
        echo [FEHLER] Junction konnte nicht erstellt werden.
        echo Bitte als Administrator ausfuehren oder manuell erstellen:
        echo   mklink /J "G:\AI\LexyAI\docs" "G:\DATEN\Obsidian\KI-Memory\KI-Memory\10-projects\lexy"
    ) else (
        echo [OK] Junction erstellt!
    )
)

echo.
echo === Lexy AI — v1 Source Junction ===
echo.

if exist "G:\AI\LexyAI\lexy_v1" (
    echo [OK] v1 Junction existiert bereits: G:\AI\LexyAI\lexy_v1
) else (
    echo Erstelle Junction zu Lexy v1 Source (Portierungs-Referenz)...
    mklink /J "G:\AI\LexyAI\lexy_v1" "G:\AI\Lexy_OS"
    if errorlevel 1 (
        echo [FEHLER] Junction konnte nicht erstellt werden.
        echo   mklink /J "G:\AI\LexyAI\lexy_v1" "G:\AI\Lexy_OS"
    ) else (
        echo [OK] v1 Junction erstellt!
    )
)

echo.
echo Fertig. Claude Code hat jetzt Zugriff auf:
echo   docs\         = Obsidian Knowledge Base (lesen + schreiben)
echo   lexy_v1\      = Lexy OS v1 Source Code (lesen, Portierungs-Referenz)
echo   referenz\     = Referenzprogramme (lesen)
echo.
pause
