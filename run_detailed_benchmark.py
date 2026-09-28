"""
Script: run_detailed_benchmark.py
Mục đích: Chạy thử nghiệm thực tế trên 30 bài hát ngẫu nhiên có thật từ dataset,
          đo lường trực tiếp từng bài hát qua 2 chế độ:
          1. Lyrics Search (Chỉ Lời — Whisper + SBERT + FAISS)
          2. Hybrid Fusion Search (Kết hợp CLAP Audio + SBERT Lyrics)
Xuất số liệu chi tiết từng bài ra JSON.
"""

import os
import time
import json
import tempfile
import soundfile as sf
import numpy as np
import librosa
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from config import (
    get_tracks_collection, PROJECT_ROOT,
    FAISS_INDEX_PATH, ID_MAPPING_PATH,
    WHISPER_MODEL, WHISPER_LANGUAGE, SBERT_MODEL
)
from hybrid_engine import search_hybrid

BASE_DIR = str(PROJECT_ROOT)
OUTPUT_JSON = os.path.join(BASE_DIR, "dataset", "detailed_track_benchmark.json")

def main():
    print("=" * 80)
    print("🚀 BẮT ĐẦU BENCHMARK THỰC NGHIỆM (CHỈ TIẾNG VIỆT)")
    print("=" * 80)

    # 1. Load models
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    whisper_model = pipeline(
        "automatic-speech-recognition", model=WHISPER_MODEL, device=device,
        generate_kwargs={"task": "transcribe", "language": WHISPER_LANGUAGE}
    )
    sbert_model = SentenceTransformer(SBERT_MODEL, device="cpu")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(str(ID_MAPPING_PATH), "r", encoding="utf-8") as f:
        chunk_mapping = json.load(f)
    tracks_col = get_tracks_collection()

    # 2. Chọn 30 bài hát tiếng Việt có cả audio snippet và lời
    query = {
        "snippet.path": {"$ne": None},
        "lyrics.full_text": {"$ne": None},
        "metadata.language": "vi",
        "local_assets.status": {"$in": ["audio_ready", "whisper_done"]}
    }
    all_tracks = list(tracks_col.find(query))
    
    step = max(1, len(all_tracks) // 30)
    sampled = all_tracks[::step][:30]
    total = len(sampled)
    print(f"Đã chọn {total} bài hát tiếng Việt để kiểm tra.")

    results_table = []
    summary = {
        "total": total,
        "lyrics": {"top1": 0, "top3": 0, "mrr": []},
        "hybrid": {"top1": 0, "top3": 0, "mrr": []}
    }

    sr = 16000
    for idx, track in enumerate(sampled, start=1):
        vid = track["_id"]
        title = track["metadata"]["title"]
        artist = ", ".join(track["metadata"].get("artist", []))
        audio_path = track["snippet"]["path"]

        if not os.path.isabs(audio_path):
            audio_path = os.path.join(BASE_DIR, audio_path)

        if not os.path.exists(audio_path):
            continue

        print(f"\n[{idx}/{total}] {title} - {artist}")

        # Cắt đoạn 8 giây (từ 5.0s đến 13.0s)
        y, _ = librosa.load(audio_path, sr=sr)
        start_s = int(sr * 5.0)
        end_s = int(sr * 13.0)
        if len(y) < end_s:
            y_clip = y[: int(sr * 8.0)]
        else:
            y_clip = y[start_s:end_s]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y_clip, sr)
            clip_path = tmp.name

        # --- A. TEST HYBRID (CLAP Audio + SBERT Lyrics) ---
        t0 = time.time()
        res_hyb, trans, msg = search_hybrid(
            clip_path, whisper_model, sbert_model, index, chunk_mapping, tracks_col, top_k=5, min_duration=5.0
        )
        lat_hyb = time.time() - t0

        rank_hyb = 0
        score_hyb = 0.0
        fusion_desc = ""
        for r_i, r in enumerate(res_hyb, 1):
            if r["video_id"] == vid:
                rank_hyb = r_i
                score_hyb = r["final_score"]
                fusion_desc = r["fusion_desc"]
                break

        # --- B. TEST LYRICS ONLY (từ nhánh Hybrid) ---
        rank_lyr = 0
        score_lyr = 0.0
        for r_i, r in enumerate(res_hyb, 1):
            if r["video_id"] == vid:
                rank_lyr = r_i
                score_lyr = r.get("lyrics_score", 0)
                break

        # Update summary
        if rank_lyr == 1: summary["lyrics"]["top1"] += 1
        if 1 <= rank_lyr <= 3: summary["lyrics"]["top3"] += 1
        summary["lyrics"]["mrr"].append(1.0 / rank_lyr if rank_lyr > 0 else 0.0)

        if rank_hyb == 1: summary["hybrid"]["top1"] += 1
        if 1 <= rank_hyb <= 3: summary["hybrid"]["top3"] += 1
        summary["hybrid"]["mrr"].append(1.0 / rank_hyb if rank_hyb > 0 else 0.0)

        record = {
            "idx": idx,
            "video_id": vid,
            "title": title,
            "artist": artist,
            "whisper_text": trans if isinstance(trans, str) else "",
            "lyrics_rank": rank_lyr,
            "lyrics_score": round(score_lyr, 4),
            "hybrid_rank": rank_hyb,
            "hybrid_score": round(score_hyb, 4),
            "hybrid_fusion_desc": fusion_desc,
            "hybrid_latency": round(lat_hyb, 3),
        }
        results_table.append(record)

        print(f"  Hybrid: rank={rank_hyb} score={score_hyb:.3f} ({lat_hyb:.2f}s)")
        print(f"  Lyrics: rank={rank_lyr} score={score_lyr:.3f}")

        os.unlink(clip_path)

    # Compute summary
    for mode in ["lyrics", "hybrid"]:
        mrr_list = summary[mode]["mrr"]
        summary[mode]["top1_acc"] = round(summary[mode]["top1"] / total, 4) if total else 0
        summary[mode]["top3_acc"] = round(summary[mode]["top3"] / total, 4) if total else 0
        summary[mode]["mrr_score"] = round(float(np.mean(mrr_list)), 4) if mrr_list else 0

    # Print summary
    print("\n" + "=" * 80)
    print("📊 KẾT QUẢ BENCHMARK TỔNG HỢP")
    print("=" * 80)
    for mode in ["lyrics", "hybrid"]:
        m = summary[mode]
        print(f"\n  {mode.upper()}:")
        print(f"    Top-1 Accuracy: {m['top1_acc']:.1%}")
        print(f"    Top-3 Accuracy: {m['top3_acc']:.1%}")
        print(f"    MRR:           {m['mrr_score']:.4f}")

    # Save
    output_data = {"summary": summary, "results": results_table}
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Đã lưu kết quả chi tiết: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
