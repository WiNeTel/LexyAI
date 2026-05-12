@echo off
:: Lexy AI - Fast Brain (Gemma 4 E4B + Vision + Audio mmproj, Port 5006)
:: RTX 3060 (GPU 0) - handles text chat, tools, classification, AND
:: serves as the STT backend for the voice_gemma4 plugin.
::
:: With --mmproj this instance accepts BOTH image inputs (vision) AND
:: audio inputs (STT) via the OpenAI /v1/chat/completions endpoint with
:: content blocks. One server, three jobs: text, vision, audio.
::
:: NOTE: if your Gemma 4 E4B GGUF ships two separate projectors (one
:: for vision, one for audio), pass both via two --mmproj flags. Some
:: llama.cpp builds only support a single --mmproj; in that case use
:: the combined multimodal projector file.

title Lexy LLM Fast :5006 [Gemma4 E4B + vision + audio]

:: ── Paths ───────────────────────────────────────────────────────────
set LLAMA_DIR=G:\AI\LexyAI\llama.cpp
set MODEL_DIR=%LLAMA_DIR%\models
set MODEL=%MODEL_DIR%\gemma-4-E4B-it-Q4_K_M.gguf

:: mmproj = multimodal projector. For E4B this handles both vision and
:: audio. Common variants:
::   mmproj-F16.gguf
::   mmproj-gemma-4-E4B.gguf
::   gemma-4-E4B-it-mmproj.gguf
set MMPROJ=%MODEL_DIR%\mmproj-gemma-4-E4B-it-BF16.gguf

:: If you have a separate audio projector (rare), set it here. Leave
:: empty if your mmproj above already contains both modalities.
set MMPROJ_AUDIO=

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

set MMPROJ_ARG=
if exist "%MMPROJ%" (
    set MMPROJ_ARG=--mmproj "%MMPROJ%"
) else (
    echo [WARN] mmproj nicht gefunden: %MMPROJ%
    echo        Server startet OHNE Vision/Audio-Support.
    echo        Falls du die mmproj-Datei hast, passe den Pfad oben an.
)

if defined MMPROJ_AUDIO (
    if exist "%MMPROJ_AUDIO%" (
        set MMPROJ_ARG=%MMPROJ_ARG% --mmproj "%MMPROJ_AUDIO%"
    )
)

echo === Starting Lexy Fast Brain ===
echo Model:    %MODEL%
echo mmproj:   %MMPROJ%
echo Endpoint: http://127.0.0.1:5006/v1
echo GPU:      RTX 3060 (main-gpu 0)
echo Context:  16384
echo Vision:   yes
echo Audio:    yes  (STT via voice_gemma4 plugin)
echo.

:: Custom Heretic-variant Jinja template (shared E4B + 26B-A4B family).
:: The root models\chat_template.jinja is the slightly newer revision
:: (adds filter_keys param) used by the E4B model. Without --jinja,
:: llama.cpp's C++ legacy parser misses the non-standard tokens
:: (<|turn>, <|channel>, <|think|>) and applies a compatibility
:: workaround that re-renders to <start_of_turn>/<end_of_turn> — not
:: what this model was trained with.
set CHAT_TEMPLATE=%MODEL_DIR%\chat_template.jinja

"%LLAMA_DIR%\llama-server.exe" ^
    --model "%MODEL%" ^
    %MMPROJ_ARG% ^
    --jinja ^
    --chat-template-file "%CHAT_TEMPLATE%" ^
    --host 127.0.0.1 ^
    --port 5006 ^
    --ctx-size 16384 ^
    --n-gpu-layers -1 ^
    --main-gpu 0 ^
    --flash-attn on ^
    --threads 6 ^
    --batch-size 512 ^
    --ubatch-size 512 ^
    --swa-full ^
    --api-key sk-lexy-local ^
    --alias gemma-4-e4b-it

pause
