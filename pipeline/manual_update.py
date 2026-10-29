import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_tracks_collection

tracks_col = get_tracks_collection()

def manual_entry():
    query = {"lyrics.full_text": None, "local_assets.status": "pending"}
    pending_tracks = list(tracks_col.find(query))
    
    if not pending_tracks:
        print("Tuyệt vời! Tất cả bài hát đã có lời.")
        return

    print(f"=== CHẾ ĐỘ NHẬP TAY CHO {len(pending_tracks)} BÀI HÁT ===")
    print("Mẹo: Mở sẵn trình duyệt để Google tìm lời.")
    print("-----------------------------------------------------")
    
    for index, track in enumerate(pending_tracks):
        title = track["metadata"]["title"]
        artist = track["metadata"]["artist"][0] if track["metadata"]["artist"] else "Không rõ"
        
        print(f"\n[{index+1}/{len(pending_tracks)}] {title} - {artist}")
        print("  -> Lựa chọn: [1] Dán lời bài hát  |  [2] Bỏ qua (Skip)  |  [3] Thoát (Exit)")
        
        choice = input("Nhập lựa chọn của bạn (1/2/3): ").strip()
        
        if choice == '3':
            print("Đã thoát chế độ nhập tay.")
            break
        elif choice == '2':
            continue
        elif choice == '1':
            print("Dán lời bài hát vào dưới đây (Sau khi dán xong, gõ chữ END ở một dòng mới rồi nhấn Enter):")
            lines = []
            while True:
                line = input()
                if line.strip() == 'END':
                    break
                lines.append(line)
                
            lyrics_text = "\n".join(lines).strip()
            
            if lyrics_text:
                tracks_col.update_one(
                    {"_id": track["_id"]},
                    {"$set": {"lyrics.full_text": lyrics_text, "local_assets.status": "has_lyrics"}}
                )
                print("  -> Đã lưu vào MongoDB thành công!")
            else:
                print("  -> Bạn chưa nhập gì, bỏ qua bài này.")
        else:
            print("Lựa chọn không hợp lệ, tự động bỏ qua.")

if __name__ == "__main__":
    manual_entry()