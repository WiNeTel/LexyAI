@echo off
:: Lexy AI – Start ChromaDB vector database
call conda activate lexyai
chroma run --host 127.0.0.1 --port 8000 --path data/memory/chroma
