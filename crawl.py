import os
import time
from ytmusicapi import YTMusic
from pymongo import MongoClient
from dotenv import load_dotenv

# 1. CẤU HÌNH KẾT NỐI MONGODB
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "lyrics_recognition")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tracks")

client = MongoClient(MONGO_URI)
tracks_col = client[DB_NAME][COLLECTION_NAME]

# 2. KHỞI TẠO YTMUSIC 
ytmusic = YTMusic()

def fetch_ytmusic_playlist(playlist_id, min_year=2019):
    print(f"  -> Đang gọi API YouTube Music...")
    
    try:
        # Tự động phân loại hàm dựa trên tiền tố của ID
        if playlist_id.startswith('RD'):
            # Dùng cho Radio / Mix động
            playlist = ytmusic.get_watch_playlist(playlistId=playlist_id, limit=100)
            tracks = playlist.get('tracks', [])
        else:
            # Dùng cho Playlist tĩnh thông thường
            playlist = ytmusic.get_playlist(playlist_id, limit=200)
            tracks = playlist.get('tracks', [])
            
    except Exception as e:
        print(f"  -> Lỗi khi đọc playlist: {e}")
        return

    inserted_count = 0
    updated_count = 0

    for track in tracks:
        video_id = track.get('videoId')
        if not video_id: 
            continue
            
        year_text = track.get('album', {}).get('year', '0') if track.get('album') else '0'
        release_year = int(year_text) if year_text.isdigit() else 0
        
        doc = {
            "metadata": {
                "title": track.get('title'),
                "artist": [artist['name'] for artist in track.get('artists', [])],
                "release_year": release_year, 
                "video_id": video_id 
            },
            "snippet": {
                "start_time": None,
                "end_time": None,
                "path": None,
                "is_processed": False
            },
            "lyrics": {
                "full_text": None,
                "synced_lrc": None
            },
            "local_assets": {
                "status": "pending" 
            }
        }
        
        # Dùng $setOnInsert để chỉ đếm những bài thực sự mới được thêm vào
        result = tracks_col.update_one(
            {"_id": video_id}, 
            {"$set": doc}, 
            upsert=True
        )
        
        if result.upserted_id:
            inserted_count += 1
        elif result.modified_count > 0:
            updated_count += 1

    print(f"  -> Hoàn thành! Thêm mới: {inserted_count} | Cập nhật: {updated_count}")

# ==========================================
# KHỐI CHẠY CHÍNH (MAIN ENTRY)
# ==========================================
if __name__ == "__main__":
    # Thay vì 1 chuỗi, ta dùng 1 mảng (list) chứa nhiều ID
    TARGET_PLAYLISTS = [
        "PLdd6FdxsIcyfw8CIufr45y8Wx5dhSxaNG"# Playlist của bạn
        
    ]
    
    print(f"=== BẮT ĐẦU CRAWL {len(TARGET_PLAYLISTS)} PLAYLISTS ===")
    
    for index, p_id in enumerate(TARGET_PLAYLISTS):
        print(f"\n[{index + 1}/{len(TARGET_PLAYLISTS)}] Xử lý Playlist: {p_id}")
        
        fetch_ytmusic_playlist(p_id)
        
        # Tạm nghỉ 2 giây trước khi cào playlist tiếp theo (trừ playlist cuối cùng)
        if index < len(TARGET_PLAYLISTS) - 1:
            print("  -> Tạm nghỉ 2 giây để tránh Rate Limit...")
            time.sleep(2)
            
    print("\n=== HOÀN TẤT CHIẾN DỊCH CRAWL ===")
    print(f"Tổng số bài hát hiện có trong Database: {tracks_col.count_documents({})}")