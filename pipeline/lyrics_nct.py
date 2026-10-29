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

def advanced_clean_title(title):
    """Cắt tỉa mạnh tay để đối phó với các tiêu đề chứa quá nhiều metadata"""
    # Xóa toàn bộ nội dung trong ngoặc đơn và ngoặc vuông
    title = re.sub(r'\(.*?\)', '', title)
    title = re.sub(r'\[.*?\]', '', title)
    # Xóa các từ khóa gây nhiễu thường gặp
    title = re.sub(r'(?i)cover|remix|beat|instrumental|official|mv|audio', '', title)
    # Xóa phần tên ca sĩ ghép sau dấu gạch ngang
    title = re.sub(r'\s+-\s+.*', '', title)
    return title.strip()

def search_nct(title):
    """Tìm kiếm trên NhacCuaTui và trả về URL bài hát"""
    query = urllib.parse.quote(title)
    search_url = f"https://www.nhaccuatui.com/tim-kiem/bai-hat?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Lấy link bài hát đầu tiên trong danh sách kết quả tìm kiếm
            song_link = soup.select_one('.sn_search_returns_list li a.name_song')
            if song_link and 'href' in song_link.attrs:
                return song_link['href']
    except Exception as e:
        pass
    return None

def get_nct_lyrics(song_url):
    """Cào text lời bài hát từ trang chi tiết của NCT"""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(song_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Lời bài hát của NCT thường nằm trong thẻ div có id 'divLyric'
            lyric_div = soup.find('div', id='divLyric')
            if lyric_div:
                # Xóa phần thông tin người đóng góp thường nằm ở cuối lời bài hát
                for br in lyric_div.find_all('br'):
                    br.replace_with('\n')
                text = lyric_div.get_text().strip()
                # Dọn dẹp khoảng trắng thừa
                text = re.sub(r'\n{3,}', '\n\n', text)
                return text
    except Exception as e:
        pass
    return None

def run_nct_sync():
    query = {"lyrics.full_text": None, "local_assets.status": "pending"}
    pending_tracks = list(tracks_col.find(query))
    
    if not pending_tracks:
        print("Tất cả bài hát đã có lời.")
        return

    print(f"=== BẮT ĐẦU VÉT ĐÁY TỪ NHACCUATUI CHO {len(pending_tracks)} BÀI HÁT ===")
    
    success_count = 0
    failed_count = 0
    
    for index, track in enumerate(pending_tracks):
        track_id = track["_id"]
        raw_title = track["metadata"]["title"]
        artist = track["metadata"]["artist"][0] if track["metadata"]["artist"] else ""
        
        # Dùng hàm cắt tỉa mạnh tay
        cleaned_title = advanced_clean_title(raw_title)
        search_query = f"{cleaned_title} {artist}".strip()
        
        print(f"\n[{index+1}/{len(pending_tracks)}] Đang tìm NCT: {cleaned_title} - {artist}")
        
        song_url = search_nct(search_query)
        
        if song_url:
            lyrics_text = get_nct_lyrics(song_url)
            if lyrics_text and len(lyrics_text) > 20: # Đảm bảo không lấy nhầm text trống
                tracks_col.update_one(
                    {"_id": track_id},
                    {
                        "$set": {
                            "lyrics.full_text": lyrics_text,
                            "local_assets.status": "has_lyrics" 
                        }
                    }
                )
                print("  -> THÀNH CÔNG!")
                success_count += 1
            else:
                print("  -> Có link nhưng không bóc tách được lời.")
                failed_count += 1
        else:
            print("  -> Bó tay. Không tìm thấy trên NhacCuaTui.")
            failed_count += 1
            
        time.sleep(1.5)

    print(f"\n=== HOÀN TẤT VÉT ĐÁY QUA NHACCUATUI ===")
    print(f"Cứu được thêm: {success_count} bài")
    print(f"Vẫn trống: {failed_count} bài")

if __name__ == "__main__":
    run_nct_sync()