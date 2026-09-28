"""
pipeline/build_audio_index.py — Xây dựng Audio Vector Index
=============================================================
Đọc tất cả file audio snippet từ dataset/snippets/,
đưa qua CLAP Audio Encoder để tạo audio embedding,
rồi nạp vào Qdrant Vector Database.

Chạy: ./venv/bin/python pipeline/build_audio_index.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import numpy as np
import torch
import librosa
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import laion_clap

from config import (
    get_tracks_collection, get_qdrant_client,
    QDRANT_COLLECTION, CLAP_MODEL,
    CLAP_SAMPLE_RATE, SNIPPETS_DIR
)

# ==========================================
# 1. KẾT NỐI DATABASE
# ==========================================
tracks_col = get_tracks_collection()

# ==========================================
_clap_model = None

def get_clap_model():
    """Tải CLAP model theo cơ chế Lazy Singleton (chỉ tải khi thực sự gọi)."""
    global _clap_model
    if _clap_model is None:
        print("Đang tải mô hình CLAP Audio Encoder...")
        _clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
        _clap_model.load_ckpt()
        print("✅ CLAP Model đã sẵn sàng!")
    return _clap_model


def encode_audio_file(audio_path):
    """
    Tạo audio embedding từ file audio bằng CLAP.
    Trả về vector numpy (512 chiều).
    """
    model = get_clap_model()
    # CLAP yêu cầu audio ở 48kHz
    audio_data, sr = librosa.load(audio_path, sr=CLAP_SAMPLE_RATE, mono=True)
    
    # CLAP cần audio dưới dạng list of numpy arrays
    audio_embed = model.get_audio_embedding_from_data(
        x=[audio_data], 
        use_tensor=False
    )
    
    return audio_embed[0]  # Trả về vector 512 chiều


# ==========================================
# 3. KHỞI TẠO QDRANT COLLECTION
# ==========================================
def setup_qdrant_collection(qdrant_client, collection_name, vector_size=512):
    """
    Tạo collection trong Qdrant nếu chưa tồn tại.
    Nếu đã tồn tại, sẽ xóa và tạo lại (rebuild index).
    """
    # Kiểm tra collection đã tồn tại chưa
    collections = [c.name for c in qdrant_client.get_collections().collections]
    
    if collection_name in collections:
        print(f"  -> Collection '{collection_name}' đã tồn tại. Xóa và tạo lại...")
        qdrant_client.delete_collection(collection_name)
    
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE  # Cosine Similarity
        )
    )
    print(f"  -> Đã tạo collection '{collection_name}' (vector_size={vector_size}, distance=COSINE)")


# ==========================================
# 4. LUỒNG CHẠY CHÍNH
# ==========================================
def build_audio_index():
    """
    Đọc tất cả audio snippet đã có trong MongoDB,
    encode bằng CLAP, rồi nạp vào Qdrant.
    """
    # Lấy tất cả bài hát đã có file audio sẵn sàng
    query = {
        "snippet.path": {"$ne": None},
        "local_assets.status": {"$in": ["audio_ready", "whisper_done"]}
    }
    tracks = list(tracks_col.find(query))
    
    print(f"\n=== BẮT ĐẦU XÂY DỰNG AUDIO INDEX CHO {len(tracks)} BÀI HÁT ===")
    
    if len(tracks) == 0:
        print("Không có bài hát nào đã sẵn sàng audio. Chạy pipeline/download_snippets.py trước!")
        return
    
    # Kết nối Qdrant
    qdrant_client = get_qdrant_client()
    setup_qdrant_collection(qdrant_client, QDRANT_COLLECTION, vector_size=512)
    
    # Encode và nạp từng bài
    points = []
    success_count = 0
    failed_count = 0
    
    for index, track in enumerate(tracks):
        track_id = track["_id"]
        title = track["metadata"]["title"]
        artists = track["metadata"].get("artist", [])
        raw_audio_path = track.get("snippet", {}).get("path") or ""
        audio_path = raw_audio_path.replace('\\', '/')
        
        # Chuyển path tương đối thành tuyệt đối
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(str(Path(__file__).resolve().parent.parent), audio_path)
            
        if not os.path.exists(audio_path):
            # Fallback sang dataset/snippets/<id>.wav
            fallback = os.path.join(str(Path(__file__).resolve().parent.parent), "dataset", "snippets", f"{track_id}.wav")
            if os.path.exists(fallback):
                audio_path = fallback
        
        print(f"  [{index+1}/{len(tracks)}] Encoding: {title}...", end="", flush=True)
        
        if not os.path.exists(audio_path):
            print(" [BỎ QUA - File không tồn tại]")
            failed_count += 1
            continue
        
        try:
            # Tạo audio embedding bằng CLAP
            embedding = encode_audio_file(audio_path)
            
            # Tạo point cho Qdrant
            point = PointStruct(
                id=index,  # Qdrant cần ID dạng int hoặc UUID
                vector=embedding.tolist(),
                payload={
                    "song_id": str(track_id),
                    "title": title,
                    "artist": ", ".join(artists) if isinstance(artists, list) else str(artists),
                    "audio_path": track["snippet"]["path"]
                }
            )
            points.append(point)
            success_count += 1
            print(" [OK]")
            
        except Exception as e:
            print(f" [LỖI: {e}]")
            failed_count += 1
    
    # Nạp batch vào Qdrant
    if points:
        print(f"\n  -> Đang nạp {len(points)} vectors vào Qdrant...")
        
        # Nạp theo batch 100 điểm
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=batch
            )
        
        print(f"  -> Đã nạp thành công {len(points)} vectors vào Qdrant!")
    
    print(f"\n=== HOÀN TẤT XÂY DỰNG AUDIO INDEX ===")
    print(f"Thành công: {success_count} | Thất bại: {failed_count}")
    
    # Kiểm tra collection
    info = qdrant_client.get_collection(QDRANT_COLLECTION)
    print(f"Qdrant Collection '{QDRANT_COLLECTION}': {info.points_count} vectors")


if __name__ == "__main__":
    build_audio_index()
