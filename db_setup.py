import os
from pymongo import MongoClient, TEXT, ASCENDING
from dotenv import load_dotenv

# ==========================================
# 1. CẤU HÌNH MÔI TRƯỜNG & KẾT NỐI
# ==========================================
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "lyrics_recognition")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tracks")

def get_database_collection():
    """Khởi tạo kết nối và trả về collection làm việc."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

# ==========================================
# 2. KHỞI TẠO INDEX (CHẠY 1 LẦN)
# ==========================================
def setup_indexes(collection):
    """Thiết lập các chỉ mục (indexes) để tối ưu truy vấn."""
    print("[1/3] Đang kiểm tra và thiết lập Indexes...")
    
    # Text Index: Phục vụ tìm kiếm bài hát siêu tốc
    collection.create_index([
        ("metadata.title", TEXT),
        ("metadata.artist", TEXT)
    ], name="text_search_index")
    
    # Status Index: Phục vụ lọc các bài hát đang chờ tải nhạc/xử lý
    collection.create_index([
        ("local_assets.status", ASCENDING)
    ], name="status_index")
    
    print("      -> Hoàn tất thiết lập Index.")

# ==========================================
# 3. HÀM TƯƠNG TÁC DỮ LIỆU (MODULE HÓA)
# ==========================================
def insert_track(collection, track_data):
    """
    Thêm một bản ghi mới với Schema chuẩn bị sẵn cho Snippet-based.
    Sử dụng upsert để không bị lỗi nếu chạy lại script nhiều lần.
    """
    doc = {
        "metadata": {
            "title": track_data.get('title'),
            "artist": track_data.get('artists', []),
            "release_date": track_data.get('release_date')
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
    
    # Dùng update_one với upsert=True để cập nhật nếu ID đã tồn tại, hoặc tạo mới
    result = collection.update_one(
        {"_id": track_data['id']}, 
        {"$set": doc}, 
        upsert=True
    )
    
    if result.upserted_id:
        print(f"      -> [Thêm mới] Document ID: {track_data['id']}")
    else:
        print(f"      -> [Cập nhật] Document ID: {track_data['id']} đã tồn tại.")

def check_database_status(collection):
    """In ra thống kê cơ bản của Database."""
    total = collection.count_documents({})
    pending = collection.count_documents({"local_assets.status": "pending"})
    print(f"\n[3/3] Thống kê Database '{DB_NAME}.{COLLECTION_NAME}':")
    print(f"      -> Tổng số bài hát: {total}")
    print(f"      -> Số bài chờ xử lý (pending): {pending}")

# ==========================================
# 4. LUỒNG CHẠY KIỂM TRA (MAIN ENTRY)
# ==========================================
if __name__ == "__main__":
    print("=== BẮT ĐẦU KHỞI TẠO HỆ THỐNG DATABASE ===\n")
    
    # 1. Kết nối
    tracks_col = get_database_collection()
    
    # 2. Tạo Index
    setup_indexes(tracks_col)
    
    # 3. Chèn thử một bản ghi mẫu (giả lập dữ liệu từ Spotify)
    print("\n[2/3] Chèn dữ liệu mẫu (Dummy Data)...")
    dummy_spotify_track = {
        "id": "test_id_sontung_123",
        "title": "Chúng Ta Của Hiện Tại",
        "artists": ["Sơn Tùng M-TP"],
        "release_date": "2020-12-20"
    }
    insert_track(tracks_col, dummy_spotify_track)
    
    # 4. Kiểm tra lại trạng thái
    check_database_status(tracks_col)
    
    print("\n=== HOÀN TẤT SETUP MONGODB KHÔNG LỖI ===")