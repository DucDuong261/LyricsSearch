from pymongo import MongoClient

def extract_all_songs():
    print("Đang kết nối vào Database...")
    client = MongoClient("mongodb://localhost:27017/")
    db = client["lyrics_recognition"]
    tracks_col = db["tracks"]

    # Chỉ truy xuất trường metadata để tăng tốc độ và giảm RAM
    tracks = tracks_col.find({"metadata": {"$exists": True}}, {"metadata.title": 1, "metadata.artist": 1, "_id": 0})

    song_list = []
    for track in tracks:
        title = track.get('metadata', {}).get('title', 'Không rõ tên')
        artists = ", ".join(track.get('metadata', {}).get('artist', ['Không rõ ca sĩ']))
        song_list.append(f"{title} - {artists}")

    # Sắp xếp danh sách theo bảng chữ cái A-Z
    song_list.sort()

    # In ra màn hình
    print(f"\n🎧 TỔNG SỐ BÀI HÁT: {len(song_list)}")
    print("="*50)
    for idx, song in enumerate(song_list, start=1):
        print(f"{idx}. {song}")
    print("="*50)

    # Lưu ra file text để dễ copy/paste tìm kiếm
    output_filename = "danh_sach_bai_hat.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"TỔNG SỐ BÀI HÁT: {len(song_list)}\n")
        f.write("="*50 + "\n")
        for idx, song in enumerate(song_list, start=1):
            f.write(f"{idx}. {song}\n")

    print(f"✅ Đã xuất thành công toàn bộ danh sách ra file: {output_filename}")

if __name__ == "__main__":
    extract_all_songs()