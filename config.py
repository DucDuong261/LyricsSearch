"""
config.py — Single Source of Truth
====================================
Module cấu hình tập trung cho toàn bộ hệ thống AI Music Matcher.

Nguyên tắc thiết kế:
  - Mọi biến môi trường, đường dẫn, tên model, hằng số đều khai báo TẠI ĐÂY.
  - Các module khác chỉ cần: from config import ...
  - MongoDB Client sử dụng Lazy Singleton Pattern (tạo 1 lần, dùng mãi mãi).
  - Qdrant Client cũng sử dụng Lazy Singleton Pattern.
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
# 2. DATABASE CONFIGURATION — MONGODB
# ==========================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "lyrics_recognition")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tracks")

# ==========================================
# 3. DATABASE CONFIGURATION — QDRANT (Audio Vectors)
# ==========================================
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "audio_embeddings")

# ==========================================
# 4. PATH CONSTANTS
# ==========================================
DATASET_DIR = PROJECT_ROOT / "dataset"
SNIPPETS_DIR = DATASET_DIR / "snippets"
FAISS_INDEX_PATH = PROJECT_ROOT / "lyrics_db.index"
ID_MAPPING_PATH = PROJECT_ROOT / "id_mapping.json"
AUDIO_INDEX_PATH = PROJECT_ROOT / "audio_db.index"  # Backup FAISS cho audio (nếu không dùng Qdrant)

# ==========================================
# 5. AI MODEL CONFIGURATION
# ==========================================
WHISPER_MODEL = "openai/whisper-small"
WHISPER_LANGUAGE = "vi"  # Ép cứng nhận diện Tiếng Việt cho Whisper
SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CLAP_MODEL = "laion/larger_clap_music_and_speech"  # Audio Encoder cho nhánh audio embedding

# ==========================================
# 6. AUDIO PROCESSING PARAMETERS
# ==========================================
SAMPLE_RATE = 16000      # Tần số lấy mẫu chuẩn cho Whisper (16kHz)
CLAP_SAMPLE_RATE = 48000 # Tần số lấy mẫu cho CLAP (48kHz)
HOP_LENGTH = 512         # Số sample giữa 2 frame liên tiếp (STFT hop)
MIN_AUDIO_DURATION = 5.0  # Thời lượng tối thiểu (giây) để hệ thống chấp nhận

# ==========================================
# 7. SEARCH PARAMETERS
# ==========================================
LYRICS_SEARCH_THRESHOLD = 0.60  # Ngưỡng tối thiểu cho kết quả tìm kiếm lyrics
LYRICS_FETCH_K = 30             # Số lượng vector FAISS lấy ra để lọc
AUDIO_SEARCH_LIMIT = 10         # Số lượng kết quả audio trả về từ Qdrant

# Trọng số Hybrid Ranking
HYBRID_WEIGHT_LYRICS = 0.55    # Trọng số nhánh Lyrics (SBERT + FAISS)
HYBRID_WEIGHT_AUDIO = 0.45     # Trọng số nhánh Audio (CLAP + Qdrant)
HYBRID_AGREEMENT_BONUS = 0.10  # Điểm thưởng khi cả 2 nhánh đồng thuận

# ==========================================
# 8. MONGODB CONNECTION — LAZY SINGLETON
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
# 9. QDRANT CONNECTION — LAZY SINGLETON
# ==========================================
_qdrant_client = None


def get_qdrant_client():
    """
    Trả về QdrantClient Singleton.
    Lần gọi đầu tiên sẽ khởi tạo connection, các lần sau trả về instance đã có.
    """
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant_client


# ==========================================
# 10. SELF-TEST (chạy trực tiếp để verify)
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("🔧 CONFIG MODULE — SELF-TEST")
    print("=" * 50)
    print(f"  PROJECT_ROOT:      {PROJECT_ROOT}")
    print(f"  MONGO_URI:         {MONGO_URI}")
    print(f"  DB_NAME:           {DB_NAME}")
    print(f"  COLLECTION_NAME:   {COLLECTION_NAME}")
    print(f"  QDRANT_HOST:       {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"  QDRANT_COLLECTION: {QDRANT_COLLECTION}")
    print(f"  FAISS_INDEX_PATH:  {FAISS_INDEX_PATH}")
    print(f"  AUDIO_INDEX_PATH:  {AUDIO_INDEX_PATH}")
    print(f"  SNIPPETS_DIR:      {SNIPPETS_DIR}")
    print(f"  WHISPER_MODEL:     {WHISPER_MODEL}")
    print(f"  WHISPER_LANGUAGE:  {WHISPER_LANGUAGE}")
    print(f"  SBERT_MODEL:       {SBERT_MODEL}")
    print(f"  CLAP_MODEL:        {CLAP_MODEL}")
    print(f"  SAMPLE_RATE:       {SAMPLE_RATE}")
    print(f"  CLAP_SAMPLE_RATE:  {CLAP_SAMPLE_RATE}")

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

    # Test Qdrant connection (optional)
    try:
        qc = get_qdrant_client()
        collections = qc.get_collections()
        print(f"  ✅ Qdrant OK — Collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"  ⚠️ Qdrant chưa sẵn sàng (bỏ qua nếu chưa start Docker): {e}")

    print("\n" + "=" * 50)
