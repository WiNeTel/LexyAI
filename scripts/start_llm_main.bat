@echo off
:: Lexy AI - Main Brain (Gemma 4 26B-A4B + Vision mmproj, Port 5005)
:: Tesla P40 (GPU 1) - 24GB VRAM, ideal fuer das grosse MoE-Modell.
::
:: With --mmproj this instance accepts image inputs via the OpenAI
:: /v1/chat/completions endpoint (content blocks with type=image_url).
:: The 26B-A4B does NOT handle audio — audio STT is served by the
:: smaller E4B instance on :5006 (see start_llm_fast.bat).

title Lexy LLM Main :5005 [Gemma4 26B-A4B + vision]

:: ── Paths ───────────────────────────────────────────────────────────
set LLAMA_DIR=G:\AI\LexyAI\llama.cpp
set MODEL_DIR=%LLAMA_DIR%\models
set MODEL=%MODEL_DIR%\gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf

:: mmproj = multimodal projector (vision encoder). If your file is
:: named differently, just change the line below. Common variants:
::   mmproj-F16.gguf
::   mmproj-gemma-4-26B-A4B.gguf
::   gemma-4-26B-A4B-it-mmproj.gguf
set MMPROJ=%MODEL_DIR%\gemma-4-26B-A4B-it-mmproj-BF16.gguf

:: ── Sanity checks ───────────────────────────────────────────────────
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
if not exist "%MMPROJ%" (
    echo [WARN] mmproj nicht gefunden: %MMPROJ%
    echo        Server startet OHNE Vision-Support.
    echo        Falls du die mmproj-Datei hast, passe den Pfad oben an.
    set MMPROJ_ARG=
) else (
    set MMPROJ_ARG=--mmproj "%MMPROJ%"
)

echo === Starting Lexy Main Brain ===
echo Model:    %MODEL%
echo mmproj:   %MMPROJ%
echo Endpoint: http://0.0.0.0:5005/v1
echo GPU:      Tesla P40 (main-gpu 1)
echo Context:  32.768
echo Vision:   yes
echo Audio:    no
echo.

"%LLAMA_DIR%\llama-server.exe" ^
    --model "%MODEL%" ^
    %MMPROJ_ARG% ^
    --host 0.0.0.0 ^
    --port 5005 ^
    --ctx-size 50000 ^
    --n-gpu-layers -1 ^
    --main-gpu 1 ^
    --threads 8 ^
    --batch-size 1024 ^
    --ubatch-size 512 ^
    --swa-full ^
    --mlock ^
    --api-key sk-lexy-local ^
    --alias gemma-4-26b-a4b-it ^
	--parallel 2

pause
