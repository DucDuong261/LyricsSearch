"""
config.py — Single Source of Truth
====================================
Module cấu hình tập trung cho toàn bộ hệ thống AI Music Matcher.

Nguyên tắc thiết kế:
  - Mọi biến môi trường, đường dẫn, tên model, hằng số đều khai báo TẠI ĐÂY.
  - Các module khác chỉ cần: from config import ...
  - MongoDB Client sử dụng Lazy Singleton Pattern (tạo 1 lần, dùng mãi mãi).
  - Luôn tìm đúng file .env nhờ PROJECT_ROOT (không phụ thuộc working directory).
"""

import os
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv

# ==========================================
# 1. PROJECT ROOT & ENVIRONMENT
# ==========================================
# Xác định thư mục gốc của project (nơi chứa file config.py này)
# Sử dụng Path.resolve() để đảm bảo đường dẫn tuyệt đối, bất kể script
# được gọi từ thư mục nào (vd: pipeline/, tests/, hoặc root).
PROJECT_ROOT = Path(__file__).resolve().parent

# Load biến môi trường từ file .env tại thư mục gốc project
load_dotenv(PROJECT_ROOT / ".env")

# ==========================================
# 2. DATABASE CONFIGURATION
# ==========================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "lyrics_recognition")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tracks")

# ==========================================
# 3. PATH CONSTANTS
# ==========================================
DATASET_DIR = PROJECT_ROOT / "dataset"
SNIPPETS_DIR = DATASET_DIR / "snippets"
FAISS_INDEX_PATH = PROJECT_ROOT / "lyrics_db.index"
ID_MAPPING_PATH = PROJECT_ROOT / "id_mapping.json"
MELODY_DB_PATH = DATASET_DIR / "melody_database.pkl"

# ==========================================
# 4. AI MODEL CONFIGURATION
# ==========================================
WHISPER_MODEL = "openai/whisper-small"
SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ==========================================
# 5. AUDIO PROCESSING PARAMETERS
# ==========================================
SAMPLE_RATE = 16000      # Tần số lấy mẫu chuẩn cho Whisper (16kHz)
HOP_LENGTH = 512         # Số sample giữa 2 frame liên tiếp (STFT hop)
MIN_AUDIO_DURATION = 5.0  # Thời lượng tối thiểu (giây) để hệ thống chấp nhận

# ==========================================
# 6. SEARCH PARAMETERS
# ==========================================
LYRICS_SEARCH_THRESHOLD = 0.60  # Ngưỡng tối thiểu cho kết quả tìm kiếm lyrics
LYRICS_FETCH_K = 30             # Số lượng vector FAISS lấy ra để lọc
HYBRID_AGREEMENT_BONUS = 0.08   # Điểm thưởng khi cả 2 nhánh đồng thuận

# ==========================================
# 7. MONGODB CONNECTION — LAZY SINGLETON
# ==========================================
# Sử dụng Singleton Pattern: chỉ tạo 1 MongoClient cho toàn bộ ứng dụng.
# Tránh tạo hàng chục connection riêng lẻ ở mỗi module (memory leak).
_mongo_client = None


def get_mongo_client():
    """
    Trả về MongoClient Singleton.
    Lần gọi đầu tiên sẽ khởi tạo connection, các lần sau trả về instance đã có.
    """
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI)
    return _mongo_client


def get_database():
    """Trả về database object."""
    return get_mongo_client()[DB_NAME]


def get_tracks_collection():
    """
    Trả về collection 'tracks' — thao tác phổ biến nhất trong toàn bộ project.
    Sử dụng: tracks_col = get_tracks_collection()
    """
    return get_database()[COLLECTION_NAME]


# ==========================================
# 8. SELF-TEST (chạy trực tiếp để verify)
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("🔧 CONFIG MODULE — SELF-TEST")
    print("=" * 50)
    print(f"  PROJECT_ROOT:     {PROJECT_ROOT}")
    print(f"  MONGO_URI:        {MONGO_URI}")
    print(f"  DB_NAME:          {DB_NAME}")
    print(f"  COLLECTION_NAME:  {COLLECTION_NAME}")
    print(f"  FAISS_INDEX_PATH: {FAISS_INDEX_PATH}")
    print(f"  MELODY_DB_PATH:   {MELODY_DB_PATH}")
    print(f"  SNIPPETS_DIR:     {SNIPPETS_DIR}")
    print(f"  WHISPER_MODEL:    {WHISPER_MODEL}")
    print(f"  SBERT_MODEL:      {SBERT_MODEL}")
    print(f"  SAMPLE_RATE:      {SAMPLE_RATE}")

    # Test MongoDB connection
    try:
        col = get_tracks_collection()
        count = col.count_documents({})
        print(f"\n  ✅ MongoDB OK — Tổng số bài hát: {count}")
    except Exception as e:
        print(f"\n  ❌ MongoDB Error: {e}")

    # Test Singleton
    client1 = get_mongo_client()
    client2 = get_mongo_client()
    assert client1 is client2, "Singleton bị tạo thêm instance!"
    print("  ✅ Singleton Pattern OK — Chỉ có 1 MongoClient duy nhất")
    print("\n" + "=" * 50)
