import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import time
import urllib.parse
from ytmusicapi import YTMusic
from config import get_tracks_collection

# 1. KẾT NỐI MONGODB (Singleton từ config.py)
tracks_col = get_tracks_collection()

# 2. KHỞI TẠO YTMUSIC 
ytmusic = YTMusic()

# ==========================================
# BỘ LỌC TIẾNG VIỆT
# ==========================================
# Danh sách ký tự có dấu đặc trưng tiếng Việt
_VN_CHARS = set('àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ')

# Danh sách từ khóa phổ biến trong tên bài hát / ca sĩ Việt Nam
_VN_KEYWORDS = [
    'official mv', 'official music video', 'lyric video',
    'anh', 'em', 'yêu', 'nhớ', 'đừng', 'người', 'một', 'không',
    'của', 'và', 'cho', 'với', 'này', 'đã', 'được', 'những',
    'bài', 'hát', 'nhạc', 'phim', 'ost',
]

# Danh sách tên ca sĩ Việt Nam phổ biến (bổ sung nếu cần)
_VN_ARTISTS = [
    'sơn tùng', 'mtp', 'soobin', 'jack', 'hoài lâm', 'erik',
    'bích phương', 'hương ly', 'phương ly', 'amee', 'justatee',
    'đen vâu', 'binz', 'karik', 'rhymastic', 'suboi',
    'min', 'bảo anh', 'chi pu', 'hoàng thùy linh', 'mỹ tâm',
    'đức phúc', 'noo phước thịnh', 'trúc nhân', 'vũ',
    'thái học', 'phan mạnh quỳnh', 'khắc việt', 'mr.siro',
    'andiez', 'hà anh tuấn', 'hiền hồ', 'orange', 'hoàng dũng',
]


def is_vietnamese_track(title, artists):
    """
    Phát hiện bài hát tiếng Việt dựa trên tên bài hát và tên ca sĩ.
    Trả về True nếu bài hát có khả năng là tiếng Việt.
    """
    # Gộp tất cả text để kiểm tra
    combined = (title + " " + " ".join(artists)).lower()
    
    # Kiểm tra 1: Có ký tự tiếng Việt có dấu không?
    vn_count = sum(1 for c in combined if c in _VN_CHARS)
    if vn_count >= 2:  # Ít nhất 2 ký tự có dấu
        return True
    
    # Kiểm tra 2: Tên ca sĩ có trong danh sách ca sĩ Việt không?
    for vn_artist in _VN_ARTISTS:
        if vn_artist in combined:
            return True
    
    return False


def extract_playlist_id(url_or_id):
    """Tự động bóc tách Playlist ID từ đường link dài hoặc trả về nguyên bản nếu đã là ID."""
    url_or_id = url_or_id.strip()
    if not url_or_id:
        return None

    if "list=" in url_or_id:
        parsed = urllib.parse.urlparse(url_or_id)
        query_params = urllib.parse.parse_qs(parsed.query)
        return query_params.get("list", [None])[0]

    return url_or_id


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
    skipped_count = 0

    for track in tracks:
        video_id = track.get('videoId')
        if not video_id: 
            continue
        
        title = track.get('title', '')
        artists = [artist['name'] for artist in track.get('artists', [])]
        
        # LỌC TIẾNG VIỆT: Bỏ qua bài hát không phải tiếng Việt
        if not is_vietnamese_track(title, artists):
            skipped_count += 1
            continue
            
        year_text = track.get('album', {}).get('year', '0') if track.get('album') else '0'
        release_year = int(year_text) if year_text.isdigit() else 0
        
        doc = {
            "metadata": {
                "title": title,
                "artist": artists,
                "release_year": release_year, 
                "video_id": video_id,
                "language": "vi"  # Đánh dấu ngôn ngữ cố định
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

    print(f"  -> Hoàn thành! Thêm mới: {inserted_count} | Cập nhật: {updated_count} | Bỏ qua (không phải tiếng Việt): {skipped_count}")

# ==========================================
# KHỐI CHẠY CHÍNH (MAIN ENTRY)
# ==========================================
if __name__ == "__main__":
    # Đọc danh sách playlist từ sources.txt (mỗi dòng 1 link hoặc 1 ID)
    txt_path = Path(__file__).resolve().parent / "sources.txt"
    if not txt_path.exists():
        print(f"LỖI: Không tìm thấy file '{txt_path.name}'!")
        print("Vui lòng tạo file này và dán các link playlist vào (mỗi link 1 dòng).")
        sys.exit(1)

    TARGET_PLAYLISTS = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            p_id = extract_playlist_id(line)
            if p_id and p_id not in TARGET_PLAYLISTS:
                TARGET_PLAYLISTS.append(p_id)

    if not TARGET_PLAYLISTS:
        print("File txt trống hoặc không chứa link hợp lệ.")
        sys.exit(1)
    
    print(f"=== BẮT ĐẦU CRAWL {len(TARGET_PLAYLISTS)} PLAYLISTS (CHỈ LỌC TIẾNG VIỆT) ===")
    
    for index, p_id in enumerate(TARGET_PLAYLISTS):
        print(f"\n[{index + 1}/{len(TARGET_PLAYLISTS)}] Xử lý Playlist: {p_id}")
        
        fetch_ytmusic_playlist(p_id)
        
        # Tạm nghỉ 2 giây trước khi cào playlist tiếp theo (trừ playlist cuối cùng)
        if index < len(TARGET_PLAYLISTS) - 1:
            print("  -> Tạm nghỉ 2 giây để tránh Rate Limit...")
            time.sleep(2)
            
    print("\n=== HOÀN TẤT CHIẾN DỊCH CRAWL ===")
    print(f"Tổng số bài hát hiện có trong Database: {tracks_col.count_documents({})}")