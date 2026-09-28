"""
Module: audio_encoder.py
========================
Audio Encoder sử dụng CLAP (Contrastive Language-Audio Pretraining)
để tạo audio embedding và tìm kiếm bài hát trong Qdrant.

Cung cấp:
  - load_clap_model()          — Load CLAP model (cached singleton)
  - encode_audio(audio_path)   — Tạo embedding từ file audio
  - search_by_audio(...)       — Tìm kiếm bài hát qua Qdrant
  - validate_audio_duration()  — Kiểm tra thời lượng audio đầu vào
"""

import os
import numpy as np
import librosa
import laion_clap

from config import (
    get_qdrant_client, get_tracks_collection,
    QDRANT_COLLECTION, CLAP_SAMPLE_RATE,
    MIN_AUDIO_DURATION, SAMPLE_RATE, AUDIO_SEARCH_LIMIT,
    PROJECT_ROOT
)

BASE_DIR = str(PROJECT_ROOT)

# ==========================================
# 1. CLAP MODEL — SINGLETON CACHE
# ==========================================
_clap_model = None


def load_clap_model():
    """
    Load CLAP model (Lazy Singleton).
    Lần gọi đầu tiên sẽ tải model (~600MB), các lần sau trả về instance đã có.
    """
    global _clap_model
    if _clap_model is None:
        print("Đang tải mô hình CLAP Audio Encoder...")
        _clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
        _clap_model.load_ckpt()
        print("✅ CLAP Audio Encoder đã sẵn sàng!")
    return _clap_model


# ==========================================
# 2. KIỂM TRA THỜI LƯỢNG AUDIO
# ==========================================
def validate_audio_duration(audio_path, min_duration=MIN_AUDIO_DURATION):
    """
    Kiểm tra độ dài file âm thanh có đạt ngưỡng tối thiểu hay không.
    Trả về: (is_valid: bool, duration: float, message: str)
    """
    try:
        duration = librosa.get_duration(path=audio_path)
    except Exception:
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            duration = float(len(y) / sr)
        except Exception as e:
            return False, 0.0, f"Lỗi đọc file âm thanh: {str(e)}"

    if duration < min_duration:
        msg = (
            f"⚠️ Đoạn âm thanh quá ngắn ({duration:.1f}s). "
            f"Vui lòng thu âm hoặc cung cấp ít nhất {min_duration:.1f} giây để AI nhận diện chính xác!"
        )
        return False, duration, msg

    return True, duration, "Hợp lệ"


# ==========================================
# 3. ENCODE AUDIO → EMBEDDING
# ==========================================
def encode_audio(audio_path):
    """
    Tạo audio embedding từ file audio bằng CLAP.
    
    Args:
        audio_path: Đường dẫn file audio (WAV, MP3)
    
    Returns:
        np.ndarray: Vector embedding 512 chiều
    """
    model = load_clap_model()
    
    # Load audio ở tần số CLAP yêu cầu (48kHz)
    audio_data, sr = librosa.load(audio_path, sr=CLAP_SAMPLE_RATE, mono=True)
    
    # Encode qua CLAP
    embedding = model.get_audio_embedding_from_data(
        x=[audio_data],
        use_tensor=False
    )
    
    return embedding[0]  # Vector 512 chiều


# ==========================================
# 4. TÌM KIẾM BÀI HÁT QUA AUDIO (QDRANT)
# ==========================================
def search_by_audio(audio_path, top_k=5, min_duration=MIN_AUDIO_DURATION):
    """
    Tìm kiếm bài hát bằng audio embedding thông qua Qdrant.
    
    Luồng: Audio file → CLAP Encoder → Embedding → Qdrant Cosine Search
    
    Args:
        audio_path: Đường dẫn file audio đầu vào
        top_k: Số lượng kết quả trả về
        min_duration: Thời lượng tối thiểu (giây)
    
    Returns:
        results: List[dict] — Danh sách bài hát khớp nhất
        msg: str — Thông báo trạng thái
    """
    # Kiểm tra thời lượng
    is_valid, duration, msg = validate_audio_duration(audio_path, min_duration)
    if not is_valid:
        return [], msg
    
    try:
        # Encode audio đầu vào
        query_embedding = encode_audio(audio_path)
        
        # Tìm kiếm trong Qdrant
        qdrant_client = get_qdrant_client()
        if hasattr(qdrant_client, "query_points"):
            query_res = qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_embedding.tolist(),
                limit=top_k
            )
            search_results = query_res.points
        else:
            search_results = qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_embedding.tolist(),
                limit=top_k
            )
        
        # Chuyển đổi kết quả
        tracks_col = get_tracks_collection()
        results = []
        
        for hit in search_results:
            payload = hit.payload
            song_id = payload.get("song_id", "")
            
            # Lấy thêm thông tin từ MongoDB
            track = tracks_col.find_one({"_id": song_id})
            
            # Xử lý đường dẫn audio
            raw_path = payload.get("audio_path", "")
            if isinstance(raw_path, str) and raw_path.strip():
                safe_path = raw_path.replace('\\', '/')
                if not os.path.isabs(safe_path):
                    safe_path = os.path.join(BASE_DIR, safe_path)
            else:
                safe_path = ""
            
            result = {
                "video_id": song_id,
                "title": payload.get("title", "Unknown"),
                "artist": payload.get("artist", "Unknown"),
                "audio_score": float(hit.score),  # Cosine similarity từ Qdrant
                "audio_path": safe_path,
                "full_lyrics": track.get("lyrics", {}).get("full_text", "") if track else ""
            }
            results.append(result)
        
        return results, f"Tìm thấy {len(results)} kết quả (Audio Encoder)"
        
    except Exception as e:
        return [], f"Lỗi Audio Encoder: {str(e)}"
