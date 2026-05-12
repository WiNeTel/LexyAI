@echo off
:: =====================================================================
:: vendor_babylon.bat
:: Downloads the three BabylonJS scripts used by the avatar-world layer
:: into frontend\static\vendor\babylon\ so Lexy can run offline / on
:: air-gapped LANs. After this you can swap the CDN tags in index.html
:: for the local vendor paths printed at the bottom.
::
:: Run from the repo root:
::     scripts\vendor_babylon.bat
:: =====================================================================
setlocal

set "TARGET=%~dp0..\frontend\static\vendor\babylon"
if not exist "%TARGET%" (
    echo Creating %TARGET%
    mkdir "%TARGET%"
)

echo.
echo === BabylonJS vendoring ===
echo Target: %TARGET%
echo.

call :fetch "https://cdn.babylonjs.com/babylon.js" "babylon.js"
call :fetch "https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js" "babylonjs.loaders.min.js"
call :fetch "https://cdn.babylonjs.com/gui/babylon.gui.min.js" "babylon.gui.min.js"

echo.
echo === Done ===
echo Replace the CDN script tags in frontend\static\index.html with:
echo.
echo     ^<script src="/static/vendor/babylon/babylon.js" defer^>^</script^>
echo     ^<script src="/static/vendor/babylon/babylonjs.loaders.min.js" defer^>^</script^>
echo     ^<script src="/static/vendor/babylon/babylon.gui.min.js" defer^>^</script^>
echo.
endlocal
exit /b 0

:fetch
set "URL=%~1"
set "NAME=%~2"
echo Fetching %NAME% ...
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri '%URL%' -OutFile '%TARGET%\%NAME%' -UseBasicParsing } catch { Write-Host 'FAILED:' $_.Exception.Message; exit 1 }"
if errorlevel 1 (
    echo   FAILED. Network blocked or CDN unreachable.
) else (
    for %%S in ("%TARGET%\%NAME%") do echo   OK ^(%%~zS bytes^)
)
exit /b 0
