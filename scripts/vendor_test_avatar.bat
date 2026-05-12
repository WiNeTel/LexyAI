@echo off
:: =====================================================================
:: vendor_test_avatar.bat
::
:: Downloads a placeholder GLB into assets\models\lexy_base.glb so the
:: avatar-world layer can render something while you wait for your
:: Ready Player Me / Avaturn / Character Creator export.
::
:: Three options — first one that downloads successfully wins:
::
::   1) Three.js Michelle (default) — Mixamo's female character with a
::      built-in idle animation. ~3.3 MB. No ARKit blendshapes but a
::      proper female skeleton so bone_animator gets jaw/eye/head.
::
::   2) Khronos CesiumMan (CC-BY) — male, slim "wax figure" with a
::      walk animation. Tiny ~490 KB, useful for smoke-testing.
::
::   3) Three.js Soldier (CC0 via Mixamo) — male mannequin with
::      animations. Same caveats — bones only, no ARKit shapes.
::
:: Run from the repo root:
::     scripts\vendor_test_avatar.bat              (= michelle, default)
::     scripts\vendor_test_avatar.bat michelle
::     scripts\vendor_test_avatar.bat cesium
::     scripts\vendor_test_avatar.bat soldier
:: =====================================================================
setlocal

set "TARGET_DIR=%~dp0..\frontend\static\avatar-world\assets\models"
set "OUT=%TARGET_DIR%\lexy_base.glb"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=michelle"

if not exist "%TARGET_DIR%" (
    echo Target directory does not exist: %TARGET_DIR%
    echo Did you run scripts\setup_junction.bat already?
    exit /b 1
)

echo.
echo === Test avatar vendoring ===
echo Mode:   %MODE%
echo Target: %OUT%
echo.

if /i "%MODE%"=="michelle" (
    set "URL=https://threejs.org/examples/models/gltf/Michelle.glb"
    set "ATTRIBUTION=Michelle (CC0 via Mixamo / three.js examples) — female"
)
if /i "%MODE%"=="cesium" (
    set "URL=https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/CesiumMan/glTF-Binary/CesiumMan.glb"
    set "ATTRIBUTION=Cesium Man (CC-BY) by Cesium / Khronos — male"
)
if /i "%MODE%"=="soldier" (
    set "URL=https://threejs.org/examples/models/gltf/Soldier.glb"
    set "ATTRIBUTION=Soldier (CC0 via Mixamo / three.js examples) — male"
)

if "%URL%"=="" (
    echo Unknown mode "%MODE%". Use: michelle ^| cesium ^| soldier
    exit /b 1
)

echo Fetching %ATTRIBUTION%
echo URL: %URL%
echo.

powershell -NoProfile -Command ^
    "try { Invoke-WebRequest -Uri '%URL%' -OutFile '%OUT%' -UseBasicParsing; Write-Host 'Saved to %OUT%' } catch { Write-Host 'FAILED:' $_.Exception.Message; exit 1 }"

if errorlevel 1 (
    echo.
    echo Download failed. Check your network or use the other option:
    echo     scripts\vendor_test_avatar.bat soldier
    echo     scripts\vendor_test_avatar.bat cesium
    exit /b 1
)

for %%S in ("%OUT%") do echo Size: %%~zS bytes
echo.
echo === Done ===
echo The avatar-world loader picks up lexy_base.glb on its first try.
echo Reload the frontend in your browser to see the new avatar.
echo.
echo Attribution: %ATTRIBUTION%
echo.
endlocal
exit /b 0
