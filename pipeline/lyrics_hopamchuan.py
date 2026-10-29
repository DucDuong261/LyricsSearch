import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import re
import requests
from bs4 import BeautifulSoup
import urllib.parse
from config import get_tracks_collection

tracks_col = get_tracks_collection()

def clean_title(title):
    """Làm sạch tiêu đề để tối ưu hóa tìm kiếm."""
    title = re.sub(r'\(.*?Official.*?\)', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\[.*?Lyric.*?\]', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\(Video\)', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\(Audio\)', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s+-\s+.*', '', title)
    return title.strip()

def search_hopamchuan(title):
    """Tìm bài hát trên hopamchuan.com và trả về URL của bài đó."""
    # Mã hóa tiêu đề để đưa vào URL
    query = urllib.parse.quote(title)
    search_url = f"https://hopamchuan.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Tìm danh sách kết quả (class 'song-item' thường chứa các bài hát)
            results = soup.find_all('a', class_='song-title', limit=1) 
            if results:
                # Trả về link trực tiếp của bài hát đầu tiên tìm thấy
                return results[0]['href']
    except Exception as e:
        pass
    return None

def get_hopamchuan_lyrics(song_url):
    """Vào trang bài hát và cào lời, lọc bỏ hợp âm."""
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        response = requests.get(song_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Lời bài hát trên Hợp Âm Chuẩn thường nằm trong pre tag hoặc div id 'song-lyric'
            lyric_div = soup.find('div', id='song-lyric')
            if lyric_div:
                raw_text = lyric_div.get_text(separator="\n")
                # Lọc bỏ các hợp âm nằm trong ngoặc vuông, ví dụ [Am], [C]
                clean_text = re.sub(r'\[.*?\]', '', raw_text)
                # Dọn dẹp khoảng trắng và dòng trống thừa
                clean_text = re.sub(r'\n\s*\n', '\n', clean_text).strip()
                return clean_text
    except Exception as e:
        pass
    return None

def run_hopamchuan_sync():
    query = {"lyrics.full_text": None, "local_assets.status": "pending"}
    pending_tracks = list(tracks_col.find(query))
    
    if not pending_tracks:
        print("Tất cả bài hát đã có lời.")
        return

    print(f"=== BẮT ĐẦU CÀO TỪ HỢP ÂM CHUẨN CHO {len(pending_tracks)} BÀI HÁT ===")
    
    success_count = 0
    failed_count = 0
    
    for index, track in enumerate(pending_tracks):
        track_id = track["_id"]
        raw_title = track["metadata"]["title"]
        artist = track["metadata"]["artist"][0] if track["metadata"]["artist"] else ""
        
        cleaned_title = clean_title(raw_title)
        print(f"\n[{index+1}/{len(pending_tracks)}] Đang tìm trên Hợp Âm Chuẩn: {cleaned_title} - {artist}")
        
        song_url = search_hopamchuan(cleaned_title)
        
        if song_url:
            lyrics_text = get_hopamchuan_lyrics(song_url)
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
                print("  -> THÀNH CÔNG! Đã lấy lời và lọc sạch hợp âm.")
                success_count += 1
            else:
                print("  -> Lỗi khi bóc tách lời từ trang.")
                failed_count += 1
        else:
            print("  -> Không tìm thấy trên Hợp Âm Chuẩn.")
            failed_count += 1
            
        # Nghỉ 2 giây để tránh làm quá tải server
        time.sleep(2)

    print(f"\n=== HOÀN TẤT VÉT ĐÁY QUA HỢP ÂM CHUẨN ===")
    print(f"Bổ sung thành công: {success_count} bài")
    print(f"Vẫn chưa có dữ liệu: {failed_count} bài")

if __name__ == "__main__":
    run_hopamchuan_sync()