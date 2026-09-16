#!/bin/bash
# Script khởi chạy ứng dụng AI Music Matcher (Streamlit Web)
cd "$(dirname "$0")"

# Đảm bảo PATH có ~/.local/bin cho ffmpeg
export PATH="$HOME/.local/bin:$PATH"

# Kích hoạt môi trường ảo
source venv/bin/activate

# Khởi chạy ứng dụng Streamlit
streamlit run demoweb.py
