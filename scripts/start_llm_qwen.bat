@echo off
:: Lexy AI - Qwen3.6 Brain (35B-A3B MoE + Vision mmproj, Port 5005)
:: Tesla P40 (GPU 1) - 24 GB VRAM. MoE mit 3 B aktiven Params -> schnell.
::
:: Belegt den A4B-Slot (Port 5005) und ersetzt damit das Gemma 26B.
:: E4B (5006) laeuft NICHT mehr -> der LLM-Client routet e4b-Requests
:: automatisch hierher (siehe llm_client._resolve fallback).

title Lexy LLM Qwen3.6 :5005 [35B-A3B + vision]

:: == Modell-Auswahl ==================================================
echo.
echo =============================================
echo   Lexy AI - Qwen3.6-35B-A3B Brain Launcher
echo =============================================
echo.
echo Welches Modell starten?
echo.
echo   [1] Qwen3.6-35B-A3B Original           (Q4_K_M, 19.7 GB)
echo   [2] OpenYourMind Qwen3.6 Uncensored    (Q4_K_S, 18.5 GB, kuato-DPO abliterated)
echo.

set CHOICE=
set /p CHOICE="Auswahl [1/2]: "

:: == Paths ===========================================================
set LLAMA_DIR=G:\AI\LexyAI\llama.cpp
set MODEL_DIR=%LLAMA_DIR%\models\qwen3.6-35b-a3b
set MMPROJ=%MODEL_DIR%\mmproj-Qwen3.6-35B-A3B-BF16.gguf
set CHAT_TEMPLATE=%MODEL_DIR%\chat_template.jinja

if "%CHOICE%"=="1" (
    set MODEL=%MODEL_DIR%\Qwen3.6-35B-A3B-Q4_K_M.gguf
    set MODEL_LABEL=Qwen3.6-35B-A3B Original
) else if "%CHOICE%"=="2" (
    set MODEL=%MODEL_DIR%\OpenYourMind-Qwen3.6-35B-A3B-kuato-DPO-abliterated-uncensored.i1-Q4_K_S.gguf
    set MODEL_LABEL=OpenYourMind Uncensored
) else (
    echo [FEHLER] Ungueltige Auswahl: "%CHOICE%"
    pause
    exit /b 1
)

:: == Sanity checks ===================================================
if not exist "%MODEL%" (
    echo [FEHLER] Modell nicht gefunden: %MODEL%
    pause
    exit /b 1
)
if not exist "%LLAMA_DIR%\llama-server.exe" (
    echo [FEHLER] llama-server.exe nicht gefunden in: %LLAMA_DIR%
    pause
    exit /b 1
)
if not exist "%CHAT_TEMPLATE%" (
    echo [FEHLER] Chat-Template nicht gefunden: %CHAT_TEMPLATE%
    pause
    exit /b 1
)

set MMPROJ_ARG=
if exist "%MMPROJ%" (
    set MMPROJ_ARG=--mmproj "%MMPROJ%"
) else (
    echo [WARN] mmproj nicht gefunden: %MMPROJ%
    echo        Server startet OHNE Vision-Support.
)

echo.
echo === Starting Lexy Qwen Brain ===
echo Modell:   %MODEL_LABEL%
echo Datei:    %MODEL%
echo mmproj:   %MMPROJ%
echo Template: %CHAT_TEMPLATE%
echo Endpoint: http://0.0.0.0:5005/v1
echo Alias:    qwen3.6-35b-a3b
echo GPU:      Tesla P40 (main-gpu 1)
echo Context:  32768
echo.

:: --split-mode none haelt das Modell komplett auf der P40, damit
:: nichts auf die 3060 Ti spillt (die ist durch User-Apps voll).
:: --cache-ram 0 verhindert RAM-KV-Cache (siehe Commit 7c125c0).
"%LLAMA_DIR%\llama-server.exe" ^
    --model "%MODEL%" ^
    %MMPROJ_ARG% ^
    --jinja ^
    --chat-template-file "%CHAT_TEMPLATE%" ^
    --host 0.0.0.0 ^
    --port 5005 ^
    --ctx-size 32768 ^
    --n-gpu-layers -1 ^
    --main-gpu 1 ^
    --split-mode none ^
    --threads 8 ^
    --batch-size 1024 ^
    --ubatch-size 512 ^
    --mlock ^
    --api-key sk-lexy-local ^
    --alias qwen3.6-35b-a3b ^
    --parallel 1 ^
    --cache-ram 0

pause
