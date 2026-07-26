import os
import re
import time
import requests
from pymongo import MongoClient
from difflib import SequenceMatcher
from dotenv import load_dotenv

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "lyrics_recognition")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tracks")

client = MongoClient(MONGO_URI)
tracks_col = client[DB_NAME][COLLECTION_NAME]

LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"

# ==========================================
# 2. CÁC HÀM BỔ TRỢ ĐỐI SOÁT DỮ LIỆU
# ==========================================
def clean_title(title):
    """Làm sạch tiêu đề bài hát từ YouTube để tăng tỷ lệ khớp API."""
    title = re.sub(r'\(.*?Official.*?\)', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\[.*?Lyric.*?\]', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\(Video\)', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\(Audio\)', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s+-\s+.*', '', title) # Cắt phần sau dấu gạch ngang nếu có
    return title.strip()

def get_similarity(a, b):
    """Tính toán độ tương đồng giữa hai chuỗi văn bản (0.0 -> 1.0)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# ==========================================
# 3. LUỒNG XỬ LÝ CHÍNH
# ==========================================
def sync_lyrics_from_lrclib():
    # Chỉ lấy các bài hát chưa có lời (lyrics.full_text là null)
    query = {"lyrics.full_text": None, "local_assets.status": "pending"}
    pending_tracks = list(tracks_col.find(query))
    
    print(f"=== BẮT ĐẦU ĐỒNG BỘ LYRICS CHO {len(pending_tracks)} BÀI HÁT ===")
    
    success_count = 0
    failed_count = 0
    
    for index, track in enumerate(pending_tracks):
        track_id = track["_id"]
        raw_title = track["metadata"]["title"]
        artists = track["metadata"]["artist"]
        
        # Lấy ca sĩ chính (phần tử đầu tiên trong mảng)
        primary_artist = artists[0] if artists else ""
        cleaned_title = clean_title(raw_title)
        
        print(f"\n[{index+1}/{len(pending_tracks)}] Đang tìm: {cleaned_title} - {primary_artist}")
        
        # Gọi API Search của LRCLIB với tham số phân rã rõ ràng
        params = {
            "track_name": cleaned_title,
            "artist_name": primary_artist
        }
        
        try:
            response = requests.get(LRCLIB_SEARCH_URL, params=params, timeout=10)
            if response.status_code != 200:
                print(f"  -> Lỗi API (Status code: {response.status_code})")
                failed_count += 1
                continue
                
            results = response.json()
            if not results:
                print("  -> Không tìm thấy lyrics trên LRCLIB.")
                failed_count += 1
                continue
            
            # Thực hiện thuật toán đối soát kết quả khớp nhất
            best_match = None
            max_score = 0.0
            
            for res in results:
                # Kiểm tra độ trùng khớp của cả tên bài lẫn tên ca sĩ
                title_score = get_similarity(cleaned_title, res.get("trackName", ""))
                artist_score = get_similarity(primary_artist, res.get("artistName", ""))
                avg_score = (title_score + artist_score) / 2
                
                if avg_score > max_score:
                    max_score = avg_score
                    best_match = res
            
            # Ngưỡng an toàn: độ khớp trung bình phải trên 70% mới lấy
            if best_match and max_score >= 0.70:
                full_text = best_match.get("plainLyrics")
                synced_lrc = best_match.get("syncedLyrics")
                
                # Nếu bài hát không có lời đồng bộ lrc thì lấy lời trơn
                if not full_text and synced_lrc:
                    # Bóc văn bản từ lrc ra nếu cần
                    full_text = re.sub(r'\[\d+:\d+\.\d+\]', '', synced_lrc).strip()
                
                if full_text:
                    # Cập nhật thông tin vào MongoDB
                    tracks_col.update_one(
                        {"_id": track_id},
                        {
                            "$set": {
                                "lyrics.full_text": full_text,
                                "lyrics.synced_lrc": synced_lrc,
                                "local_assets.status": "has_lyrics" # Chuyển trạng thái để luồng sau biết bài này đã sẵn sàng
                            }
                        }
                    )
                    print(f"  -> ĐỒNG BỘ THÀNH CÔNG! (Độ khớp: {max_score:.2f})")
                    success_count += 1
                else:
                    print("  -> Bài hát này trên LRCLIB là nhạc không lời (Instrumental).")
                    failed_count += 1
            else:
                print(f"  -> Bị loại bỏ! Kết quả tìm thấy không đáng tin cậy (Độ khớp cao nhất: {max_score:.2f})")
                failed_count += 1
                
        except Exception as e:
            print(f"  -> Xảy ra lỗi khi kết nối: {e}")
            failed_count += 1
            
        # Delay 1 giây giữa các bài hát để tôn trọng rate limit của LRCLIB (API miễn phí)
        time.sleep(1)
        
    print(f"\n=== CHIẾN DỊCH HOÀN TẤT ===")
    print(f"Tổng số bài khớp lyrics thành công: {success_count}")
    print(f"Tổng số bài không lấy được lyrics: {failed_count}")

if __name__ == "__main__":
    sync_lyrics_from_lrclib()