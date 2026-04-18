@echo off
:: Lexy AI – Startup Script
cd /d "%~dp0.."
call conda activate lexyai
python -m lexy_core
pause