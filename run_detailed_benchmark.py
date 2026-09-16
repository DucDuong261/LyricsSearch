"""
Script: run_detailed_benchmark.py
Mục đích: Chạy thử nghiệm thực tế 100% trên 30 bài hát ngẫu nhiên có thật từ dataset,
          đo lường trực tiếp từng bài hát qua 3 chế độ:
          1. Lyrics Search (Chỉ Lời)
          2. Melody QBH Search (Chỉ Giai điệu)
          3. Hybrid Fusion Search (Dung hợp cả hai)
Xuất số liệu chi tiết từng bài ra JSON và bảng Markdown.
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
    WHISPER_MODEL, SBERT_MODEL
)
from melody_engine import search_by_melody, load_melody_database
from hybrid_engine import search_hybrid

BASE_DIR = str(PROJECT_ROOT)
OUTPUT_JSON = os.path.join(BASE_DIR, "dataset", "detailed_track_benchmark.json")

def main():
    print("=" * 80)
    print("🚀 BẮT ĐẦU BENCHMARK THỰC NGHIỆM CHI TIẾT TRÊN 30 BÀI HÁT")
    print("=" * 80)

    # 1. Load models
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    whisper_model = pipeline("automatic-speech-recognition", model=WHISPER_MODEL, device=device, generate_kwargs={"task": "transcribe"})
    sbert_model = SentenceTransformer(SBERT_MODEL, device="cpu")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(str(ID_MAPPING_PATH), "r", encoding="utf-8") as f:
        chunk_mapping = json.load(f)
    tracks_col = get_tracks_collection()

    # Load melody database
    mel_db = load_melody_database()
    print(f"Loaded melody DB: {len(mel_db)} bài hát.")

    # 2. Chọn 30 bài hát mẫu có cả audio snippet và lời trong DB
    step = max(1, len(mel_db) // 30)
    sampled = mel_db[::step][:30]
    total = len(sampled)
    print(f"Đã chọn {total} bài hát thực tế để kiểm tra.")

    results_table = []
    summary = {
        "total": total,
        "lyrics": {"top1": 0, "top3": 0, "mrr": []},
        "melody": {"top1": 0, "top3": 0, "mrr": []},
        "hybrid": {"top1": 0, "top3": 0, "mrr": []}
    }

    sr = 16000
    for idx, entry in enumerate(sampled, start=1):
        vid = entry["video_id"]
        title = entry["title"]
        artist = entry["artist"]
        audio_path = entry["audio_path"]

        if not os.path.exists(audio_path):
            continue

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

        # --- A. TEST MELODY ONLY ---
        t0 = time.time()
        res_mel, _, _ = search_by_melody(clip_path, top_k=5, min_duration=5.0)
        lat_mel = time.time() - t0
        rank_mel = 0
        score_mel = 0.0
        for r_i, r in enumerate(res_mel, 1):
            if r["video_id"] == vid:
                rank_mel = r_i
                score_mel = r["score"]
                break

        # --- B. TEST HYBRID (CHỨA CẢ NHÁNH LỜI VÀ GIAI ĐIỆU) ---
        t0 = time.time()
        res_hyb, trans, _, _ = search_hybrid(
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

        # --- C. TEST LYRICS ONLY ---
        # Trích xuất riêng rank từ nhánh Lời
        rank_lyr = 0
        score_lyr = 0.0
        if trans and len(trans.strip()) >= 5 and trans != "Nhiễu/Không nghe rõ":
            q_vec = sbert_model.encode([trans], normalize_embeddings=True)
            d, ind = index.search(np.array(q_vec).astype("float32"), 30)
            seen = set()
            lyr_ranked = []
            for midx in ind[0]:
                mid = chunk_mapping[midx]["mongo_id"]
                if mid not in seen:
                    seen.add(mid)
                    lyr_ranked.append(mid)
            for r_i, mid in enumerate(lyr_ranked, 1):
                if mid == vid:
                    rank_lyr = r_i
                    score_lyr = float(d[0][r_i - 1]) if r_i - 1 < len(d[0]) else 0.5
                    break

        os.remove(clip_path)

        # Cập nhật thống kê
        if rank_lyr == 1: summary["lyrics"]["top1"] += 1
        if 1 <= rank_lyr <= 3: summary["lyrics"]["top3"] += 1
        summary["lyrics"]["mrr"].append(1.0 / rank_lyr if rank_lyr > 0 else 0.0)

        if rank_mel == 1: summary["melody"]["top1"] += 1
        if 1 <= rank_mel <= 3: summary["melody"]["top3"] += 1
        summary["melody"]["mrr"].append(1.0 / rank_mel if rank_mel > 0 else 0.0)

        if rank_hyb == 1: summary["hybrid"]["top1"] += 1
        if 1 <= rank_hyb <= 3: summary["hybrid"]["top3"] += 1
        summary["hybrid"]["mrr"].append(1.0 / rank_hyb if rank_hyb > 0 else 0.0)

        row = {
            "index": idx,
            "video_id": vid,
            "title": title,
            "artist": artist,
            "whisper_transcription": trans[:40] + "..." if len(trans) > 40 else trans,
            "lyrics_rank": rank_lyr if rank_lyr > 0 else "Miss",
            "melody_rank": rank_mel if rank_mel > 0 else "Miss",
            "melody_score": round(score_mel * 100, 1),
            "hybrid_rank": rank_hyb if rank_hyb > 0 else "Miss",
            "hybrid_score": round(score_hyb * 100, 1),
            "fusion_desc": fusion_desc
        }
        results_table.append(row)

        print(f"[{idx:02d}/{total}] {title[:25]:<25} | Lời: #{str(row['lyrics_rank']):<4} | Giai điệu: #{str(row['melody_rank']):<4} | Hybrid: #{str(row['hybrid_rank']):<4} ({row['hybrid_score']}%)")

    # Tính tổng kết
    summary["lyrics"]["top1_acc"] = round(summary["lyrics"]["top1"] / total * 100, 1)
    summary["lyrics"]["top3_acc"] = round(summary["lyrics"]["top3"] / total * 100, 1)
    summary["lyrics"]["mrr_score"] = round(float(np.mean(summary["lyrics"]["mrr"])), 4)

    summary["melody"]["top1_acc"] = round(summary["melody"]["top1"] / total * 100, 1)
    summary["melody"]["top3_acc"] = round(summary["melody"]["top3"] / total * 100, 1)
    summary["melody"]["mrr_score"] = round(float(np.mean(summary["melody"]["mrr"])), 4)

    summary["hybrid"]["top1_acc"] = round(summary["hybrid"]["top1"] / total * 100, 1)
    summary["hybrid"]["top3_acc"] = round(summary["hybrid"]["top3"] / total * 100, 1)
    summary["hybrid"]["mrr_score"] = round(float(np.mean(summary["hybrid"]["mrr"])), 4)

    output_data = {
        "summary": summary,
        "detailed_tracks": results_table
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("📊 TỔNG KẾT ĐỐI ĐẦU TRỰC TIẾP TRÊN 30 BÀI HÁT THỰC TẾ")
    print("=" * 80)
    print(f"1. Nhánh Lời hát (Whisper ASR):   Top-1 = {summary['lyrics']['top1']}/{total} ({summary['lyrics']['top1_acc']}%) | Top-3 = {summary['lyrics']['top3']}/{total} ({summary['lyrics']['top3_acc']}%) | MRR = {summary['lyrics']['mrr_score']}")
    print(f"2. Nhánh Giai điệu (QBH DTW):    Top-1 = {summary['melody']['top1']}/{total} ({summary['melody']['top1_acc']}%) | Top-3 = {summary['melody']['top3']}/{total} ({summary['melody']['top3_acc']}%) | MRR = {summary['melody']['mrr_score']}")
    print(f"3. Nhánh Dung hợp (Hybrid):      Top-1 = {summary['hybrid']['top1']}/{total} ({summary['hybrid']['top1_acc']}%) | Top-3 = {summary['hybrid']['top3']}/{total} ({summary['hybrid']['top3_acc']}%) | MRR = {summary['hybrid']['mrr_score']}")
    print(f"💾 File log chi tiết đã lưu: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
