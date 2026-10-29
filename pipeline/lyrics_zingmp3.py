import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import re
import requests
from config import get_tracks_collection

tracks_col = get_tracks_collection()

def clean_title(title):
    """Làm sạch tiêu đề để tăng tỷ lệ tìm kiếm chính xác."""
    title = re.sub(r'\(.*?Official.*?\)', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\[.*?Lyric.*?\]', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\(Video\)', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\(Audio\)', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s+-\s+.*', '', title)
    return title.strip()

def search_zingmp3(title, artist):
    """Tìm kiếm bài hát qua API của Zing MP3 và trả về ID bài hát."""
    search_url = "https://ac.zingmp3.vn/v1/web/ac-suggestions"
    params = {"query": f"{title} {artist}"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        response = requests.get(search_url, params=params, headers=headers, timeout=10)
        data = response.json()
        if data.get("result") and data["result"].get("items"):
            for item in data["result"]["items"]:
                # Tìm object chứa danh sách bài hát
                if item.get("id") == "Song":
                    songs = item.get("song", [])
                    if songs:
                        return songs[0].get("id") # Trả về ID của bài hát khớp nhất
    except Exception as e:
        pass
    return None

def get_zingmp3_lyrics(song_id):
    """Lấy lời bài hát từ Zing MP3 dựa trên Song ID."""
    lyrics_url = "https://zingmp3.vn/api/song/get-lyric"
    params = {"id": song_id}
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(lyrics_url, params=params, headers=headers, timeout=10)
        data = response.json()
        if data.get("err") == 0 and data.get("data"):
            # Lấy danh sách các câu hát (sentences) và ghép lại thành full_text
            sentences = data["data"].get("sentences", [])
            full_text = "\n".join([" ".join([word["data"] for word in sentence["words"]]) for sentence in sentences])
            return full_text
    except Exception as e:
        pass
    return None

def run_zingmp3_sync():
    # CHỈ LỌC NHỮNG BÀI HÁT CHƯA CÓ LỜI VÀ ĐANG CHỜ XỬ LÝ
    query = {"lyrics.full_text": None, "local_assets.status": "pending"}
    pending_tracks = list(tracks_col.find(query))
    
    if not pending_tracks:
        print("Tất cả bài hát đã có lời. Không cần chạy Zing MP3.")
        return

    print(f"=== BẮT ĐẦU CÀO LYRICS TỪ ZING MP3 CHO {len(pending_tracks)} BÀI HÁT ===")
    
    success_count = 0
    failed_count = 0
    
    for index, track in enumerate(pending_tracks):
        track_id = track["_id"]
        raw_title = track["metadata"]["title"]
        artist = track["metadata"]["artist"][0] if track["metadata"]["artist"] else ""
        
        cleaned_title = clean_title(raw_title)
        print(f"\n[{index+1}/{len(pending_tracks)}] Đang tìm trên Zing: {cleaned_title} - {artist}")
        
        # 1. Tìm ID bài hát trên Zing
        zing_id = search_zingmp3(cleaned_title, artist)
        
        if zing_id:
            # 2. Dùng ID lấy lời bài hát
            lyrics_text = get_zingmp3_lyrics(zing_id)
            
            if lyrics_text:
                tracks_col.update_one(
                    {"_id": track_id},
                    {
                        "$set": {
                            "lyrics.full_text": lyrics_text,
                            "local_assets.status": "has_lyrics" 
                        }
                    }
                )
                print("  -> THÀNH CÔNG! Đã lấy lời từ Zing MP3.")
                success_count += 1
            else:
                print("  -> Zing MP3 có bài hát nhưng hệ thống báo trống lời.")
                failed_count += 1
        else:
            print("  -> Không tìm thấy bài hát trên Zing MP3.")
            failed_count += 1
            
        time.sleep(1) # Tránh bị chặn IP

    print(f"\n=== HOÀN TẤT VÉT ĐÁY QUA ZING MP3 ===")
    print(f"Bổ sung thành công: {success_count} bài")
    print(f"Vẫn chưa có dữ liệu: {failed_count} bài")

if __name__ == "__main__":
    run_zingmp3_sync()