#!/bin/bash
# Khởi chạy giao diện web mới (SPA + API Starlette) tại http://localhost:8000
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
if [ -f venv/Scripts/activate ]; then source venv/Scripts/activate; else source venv/bin/activate; fi
python -m uvicorn webapp.server:app --host 127.0.0.1 --port 8000
