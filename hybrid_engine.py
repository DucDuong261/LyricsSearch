"""
Module: hybrid_engine.py
Mục đích: Động cơ tìm kiếm âm nhạc thông minh lai (Multimodal Hybrid Fusion)
Kết hợp đồng thời 2 luồng nhận diện:
  1. Nhánh Ngữ âm (Whisper ASR + SBERT + FAISS)
  2. Nhánh Giai điệu (Chroma CENS + F0 Tracking + Subsequence DTW)

Chạy song song bằng đa luồng (ThreadPoolExecutor) để tối ưu thời gian phản hồi.
Cơ chế Dung hợp Điểm số Thích ứng (Adaptive Score Fusion) & Đồng thuận (Agreement Bonus).
"""

import os
import time
import concurrent.futures
import numpy as np
import librosa
from melody_engine import (
    validate_audio_duration,
    search_by_melody,
    MIN_AUDIO_DURATION
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_lyrics_branch(audio_path, whisper_model, sbert_model, index, chunk_mapping, tracks_col, top_k=15):
    """
    Nhánh 1: Nhận diện tiếng nói (Whisper) -> Vector ngữ nghĩa (SBERT) -> FAISS
    """
    try:
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
        result = whisper_model({"array": audio_array, "sampling_rate": sampling_rate})
        text = result["text"].strip()

        if not text or len(text) < 5:
            return [], text or "Nhiễu/Không nghe rõ"

        # SBERT encoding
        query_vector = sbert_model.encode([text], normalize_embeddings=True)
        fetch_k = min(40, index.ntotal)
        distances, indices = index.search(np.array(query_vector).astype("float32"), fetch_k)

        seen_ids = set()
        seen_songs = set()
        candidates = []

        for i in range(fetch_k):
            if len(candidates) >= top_k:
                break
            match_info = chunk_mapping[indices[0][i]]
            mongo_id = match_info["mongo_id"]
            if mongo_id in seen_ids:
                continue
            seen_ids.add(mongo_id)

            track = tracks_col.find_one({"_id": mongo_id})
            if track:
                title = track.get("metadata", {}).get("title", mongo_id)
                artist_list = track.get("metadata", {}).get("artist", ["Unknown"])
                artist_str = ", ".join(artist_list) if isinstance(artist_list, list) else str(artist_list)

                sig = (title.strip().lower(), artist_str.strip().lower())
                if sig in seen_songs:
                    continue
                seen_songs.add(sig)

                raw_path = track.get("snippet", {}).get("path", "")
                if isinstance(raw_path, str) and raw_path.strip():
                    safe_path = raw_path.replace("\\", "/")
                    if not os.path.isabs(safe_path):
                        safe_path = os.path.join(BASE_DIR, safe_path)
                else:
                    safe_path = ""

                candidates.append({
                    "video_id": mongo_id,
                    "title": title,
                    "artist": artist_str,
                    "matched_line": match_info.get("matched_line", ""),
                    "lyrics_score": float(distances[0][i]),
                    "audio_path": safe_path,
                    "full_lyrics": track.get("lyrics", {}).get("full_text", "")
                })

        return candidates, text
    except Exception as e:
        return [], f"Lỗi nhánh lời: {str(e)}"


def _run_melody_branch(audio_path, top_k=15):
    """
    Nhánh 2: Nhận diện đường cong cao độ và hòa âm (Chroma CENS + DTW)
    """
    try:
        results, query_feats, msg = search_by_melody(audio_path, top_k=top_k, min_duration=MIN_AUDIO_DURATION)
        return results, query_feats, msg
    except Exception as e:
        return [], None, f"Lỗi nhánh giai điệu: {str(e)}"


def search_hybrid(audio_path, whisper_model, sbert_model, index, chunk_mapping, tracks_col, top_k=3, min_duration=MIN_AUDIO_DURATION):
    """
    Động cơ tìm kiếm thông minh kết hợp cả 2 nhánh (Hybrid Fusion):
    1. Kiểm tra thời lượng đầu vào >= min_duration
    2. Chạy song song 2 nhánh qua ThreadPoolExecutor
    3. Dung hợp điểm số thích ứng (Adaptive Fusion) và cộng điểm đồng thuận
    4. Trả về kết quả tối ưu nhất kèm bằng chứng của cả 2 nhánh
    """
    # 0. Kiểm tra ngưỡng thời lượng
    is_valid, duration, msg = validate_audio_duration(audio_path, min_duration)
    if not is_valid:
        return [], None, None, msg

    t_start = time.time()

    # 1. Chạy song song 2 nhánh
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_lyrics = executor.submit(
            _run_lyrics_branch, audio_path, whisper_model, sbert_model, index, chunk_mapping, tracks_col, top_k=15
        )
        future_melody = executor.submit(
            _run_melody_branch, audio_path, top_k=15
        )

        lyrics_candidates, transcription = future_lyrics.result()
        melody_candidates, query_feats, melody_msg = future_melody.result()

    elapsed = time.time() - t_start

    # 2. Xác định trọng số thích ứng (Adaptive Weights)
    has_valid_lyrics = (
        isinstance(transcription, str)
        and len(transcription.strip()) >= 5
        and transcription != "Nhiễu/Không nghe rõ"
        and not transcription.startswith("Lỗi")
    )

    if has_valid_lyrics:
        w_lyrics = 0.55
        w_melody = 0.45
    else:
        # Nếu Whisper không nghe rõ hoặc chỉ là tiếng ngân nga không lời
        w_lyrics = 0.0
        w_melody = 1.0

    # 3. Gom và dung hợp kết quả theo video_id
    combined_dict = {}

    # Nạp nhánh Lời
    for item in lyrics_candidates:
        vid = item["video_id"]
        combined_dict[vid] = {
            "video_id": vid,
            "title": item["title"],
            "artist": item["artist"],
            "audio_path": item["audio_path"],
            "full_lyrics": item.get("full_lyrics", ""),
            "matched_line": item.get("matched_line", ""),
            "lyrics_score": item["lyrics_score"],
            "melody_score": 0.0,
            "matched_start_sec": None,
            "matched_end_sec": None,
            "dtw_cost": None,
            "ref_entry": None,
            "source": "lyrics_only"
        }

    # Nạp và dung hợp nhánh Giai điệu
    for item in melody_candidates:
        vid = item["video_id"]
        if vid in combined_dict:
            # Bài hát xuất hiện ở CẢ 2 NHÁNH -> CỰC KỲ ĐÁNG TIN CẬY
            entry = combined_dict[vid]
            entry["melody_score"] = item["score"]
            entry["matched_start_sec"] = item["matched_start_sec"]
            entry["matched_end_sec"] = item["matched_end_sec"]
            entry["matched_start_frame"] = item["matched_start_frame"]
            entry["matched_end_frame"] = item["matched_end_frame"]
            entry["dtw_cost"] = item["dtw_cost"]
            entry["ref_entry"] = item["ref_entry"]
            entry["source"] = "both"
            if not entry["audio_path"]:
                entry["audio_path"] = item["audio_path"]
        else:
            combined_dict[vid] = {
                "video_id": vid,
                "title": item["title"],
                "artist": item["artist"],
                "audio_path": item["audio_path"],
                "full_lyrics": "",
                "matched_line": "",
                "lyrics_score": 0.0,
                "melody_score": item["score"],
                "matched_start_sec": item["matched_start_sec"],
                "matched_end_sec": item["matched_end_sec"],
                "matched_start_frame": item["matched_start_frame"],
                "matched_end_frame": item["matched_end_frame"],
                "dtw_cost": item["dtw_cost"],
                "ref_entry": item["ref_entry"],
                "source": "melody_only"
            }

    # 4. Tính điểm tổng hợp (Final Fusion Score)
    fused_results = []
    for vid, item in combined_dict.items():
        s_lyric = item["lyrics_score"]
        s_melody = item["melody_score"]

        # Điểm thưởng đồng thuận khi cả 2 nhánh cùng chỉ về 1 bài
        agreement_bonus = 0.08 if item["source"] == "both" else 0.0

        if item["source"] == "both":
            final_score = (w_lyrics * s_lyric) + (w_melody * s_melody) + agreement_bonus
            fusion_desc = f"Khớp cả Lời ({s_lyric*100:.1f}%) & Giai điệu ({s_melody*100:.1f}%)"
        elif item["source"] == "lyrics_only":
            final_score = s_lyric * 0.9  # Giảm nhẹ nếu giai điệu không khớp
            fusion_desc = f"Khớp theo Lời bài hát ({s_lyric*100:.1f}%)"
        else:
            final_score = s_melody * 0.95
            fusion_desc = f"Khớp theo Giai điệu ({s_melody*100:.1f}%)"

        item["final_score"] = float(np.clip(final_score, 0.0, 1.0))
        item["score"] = item["final_score"]
        item["fusion_desc"] = fusion_desc
        fused_results.append(item)

    # Sắp xếp theo điểm tổng hợp giảm dần
    fused_results.sort(key=lambda x: x["final_score"], reverse=True)

    return fused_results[:top_k], transcription, query_feats, f"Xử lý thành công ({elapsed:.2f}s)"
