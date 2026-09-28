import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import subprocess
from config import get_tracks_collection, SNIPPETS_DIR

# ==========================================
# CẤU HÌNH HỆ THỐNG (Singleton từ config.py)
# ==========================================
tracks_col = get_tracks_collection()

# Thư mục lưu trữ file audio cắt ra
OUTPUT_DIR = str(SNIPPETS_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# HÀM XỬ LÝ TẢI VÀ CẮT YOUTUBE
# ==========================================
def get_ytdlp_bin() -> str:
    import shutil
    venv_bin = Path(sys.executable).parent / "yt-dlp"
    if venv_bin.exists():
        return str(venv_bin)
    which_bin = shutil.which("yt-dlp")
    if which_bin:
        return which_bin
    return "yt-dlp"

def download_audio_snippet(video_id, output_path, start_time="01:00", end_time="01:30"):
    """
    Sử dụng yt-dlp gọi ffmpeg để tải trực tiếp MỘT ĐOẠN 30s của video
    mà không cần tải toàn bộ file gốc. Ép kiểu về WAV 16kHz, Mono (chuẩn cho Whisper).
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    ytdlp_bin = get_ytdlp_bin()
    
    # Cấu trúc lệnh yt-dlp
    command = [
        ytdlp_bin,
        "--quiet", "--no-warnings",
        "-f", "bestaudio",
        "--download-sections", f"*{start_time}-{end_time}",
        "--force-keyframes-at-cuts",
        "-x", "--audio-format", "wav",
        "--postprocessor-args", "-ar 16000 -ac 1", # 16kHz, 1 channel (Mono)
        "-o", output_path,
        url
    ]
    
    try:
        # Bỏ DEVNULL, dùng capture_output để bắt lỗi thật
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        # In ra màn hình dòng chữ đỏ nguyên nhân lỗi từ yt-dlp/ffmpeg
        print(f"\n   -> [CHI TIẾT LỖI]: {e.stderr.strip()}") 
        return False    

# ==========================================
# LUỒNG CHẠY CHÍNH
# ==========================================
def process_snippets():
    # Chỉ lấy các bài hát ĐÃ CÓ LYRICS và CHƯA TẢI AUDIO
    query = {"local_assets.status": "has_lyrics"}
    valid_tracks = list(tracks_col.find(query))
    
    print(f"=== BẮT ĐẦU TẢI AUDIO SNIPPET CHO {len(valid_tracks)} BÀI HÁT ===")
    
    success_count = 0
    failed_count = 0
    
    for index, track in enumerate(valid_tracks):
        video_id = track["metadata"]["video_id"]
        title = track["metadata"]["title"]
        
        # Đường dẫn file đầu ra
        file_name = f"{video_id}.wav"
        output_path = os.path.join(OUTPUT_DIR, file_name)
        
        print(f"[{index+1}/{len(valid_tracks)}] Đang xử lý: {title}...", end="", flush=True)
        
        # Bỏ qua nếu file đã tồn tại trên ổ cứng
        if os.path.exists(output_path):
            print(" [BỎ QUA - Đã có file]")
            success_count += 1
            # Cập nhật DB phòng trường hợp script bị ngắt giữa chừng lần trước
            tracks_col.update_one(
                {"_id": video_id},
                {"$set": {
                    "snippet.path": output_path,
                    "snippet.start_time": 60,
                    "snippet.end_time": 90,
                    "local_assets.status": "audio_ready"
                }}
            )
            continue
            
        # Gọi hàm tải snippet
        is_success = download_audio_snippet(video_id, output_path)
        
        if is_success and os.path.exists(output_path):
            print(" [THÀNH CÔNG]")
            # Cập nhật trạng thái vào MongoDB
            tracks_col.update_one(
                {"_id": video_id},
                {"$set": {
                    "snippet.path": output_path,
                    "snippet.start_time": 60,
                    "snippet.end_time": 90,
                    "local_assets.status": "audio_ready"
                }}
            )
            success_count += 1
        else:
            print(" [LỖI - Bỏ qua]")
            failed_count += 1

    print(f"\n=== CHIẾN DỊCH HOÀN TẤT ===")
    print(f"Số file âm thanh tải thành công: {success_count}")
    print(f"Số file bị lỗi: {failed_count}")

if __name__ == "__main__":
    process_snippets()