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

:: Custom Heretic-variant Jinja template. The GGUF's embedded template
:: uses non-standard tokens (<|turn>, <|channel>, <|think|>) that
:: llama.cpp's C++ legacy parser doesn't understand — it falls back
:: to a "compatibility workaround" that re-renders into
:: <start_of_turn>...<end_of_turn>, which is NOT what this model was
:: trained with. Pointing --jinja at the model's own
:: chat_template.jinja makes llama.cpp render with the exact tokens
:: the weights expect. Big consistency win for narrative quality.
set CHAT_TEMPLATE=%MODEL_DIR%\gemma-4-26B-A4B\chat_template.jinja

:: --split-mode none keeps the entire 26B-A4B on the main-gpu (P40)
:: instead of letting llama.cpp's default 'layer' split spill ~5 GB
:: of tensors onto the RTX 3060. The 3060 is reserved for the E4B
:: fast brain on :5006; without this flag the main brain ate ~9.5
:: of the 12 GB on that GPU, leaving E4B no room.
::
:: ctx 32768 + parallel 1 fit on the P40 (24 GB) with comfortable
:: headroom: 16 GB tensors + 6.6 GB KV cache (single slot) + 1.3 GB
:: compute = ~24 GB ceiling. parallel=2 doubled the KV → OOM. RP
:: rounds rarely overlap a user turn; llama.cpp queues the second
:: call with maybe 1-2 s of extra latency, which is fine.
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
    --swa-full ^
    --mlock ^
    --api-key sk-lexy-local ^
    --alias gemma-4-26b-a4b-it ^
    --parallel 1 ^
    --cache-ram 0

pause
