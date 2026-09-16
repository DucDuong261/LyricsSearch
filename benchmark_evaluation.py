"""
Module: benchmark_evaluation.py
Mục đích: Đánh giá có hệ thống độ chính xác (Accuracy), MRR và Latency của 
          cả 2 động cơ tìm kiếm:
          1. Giai điệu ngân nga (Query by Humming - QBH) dưới 3 điều kiện thực tế:
             - Chuẩn (Clean 8s snippet)
             - Lệch tông (+2 semitones Pitch Shift - người dùng hát cao hơn bản gốc)
             - Nhiễu môi trường (Noisy SNR 15dB - thu âm ở nơi ồn ào)
          2. Nhận diện Lời bài hát (Whisper ASR + SBERT + FAISS) trên toàn bộ dataset
"""

import os
import time
import json
import tempfile
import soundfile as sf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import faiss
from sentence_transformers import SentenceTransformer
from config import (
    PROJECT_ROOT, SBERT_MODEL,
    FAISS_INDEX_PATH, ID_MAPPING_PATH
)
from melody_engine import (
    search_by_melody,
    load_melody_database,
    DEFAULT_DB_PATH
)

BASE_DIR = str(PROJECT_ROOT)
BENCHMARK_JSON = os.path.join(BASE_DIR, "dataset", "accuracy_benchmark_results.json")
BENCHMARK_CHART = os.path.join(BASE_DIR, "dataset", "accuracy_benchmark.png")


def evaluate_qbh_accuracy(num_samples=50):
    print("=" * 60)
    print(f"🎵 1. ĐÁNH GIÁ ĐỘ CHÍNH XÁC QUERY BY HUMMING (QBH) - N={num_samples}")
    print("=" * 60)

    db = load_melody_database(DEFAULT_DB_PATH)
    if not db:
        print("❌ Lỗi: Cơ sở dữ liệu melody_database rỗng!")
        return {}

    # Chọn num_samples bài hát đại diện
    step = max(1, len(db) // num_samples)
    sampled_entries = db[::step][:num_samples]
    total_eval = len(sampled_entries)
    print(f"📊 Đang đánh giá trên {total_eval} bài hát mẫu...")

    conditions = {
        "clean": {"hits_top1": 0, "hits_top3": 0, "reciprocal_ranks": [], "latencies": []},
        "pitch_shift_2": {"hits_top1": 0, "hits_top3": 0, "reciprocal_ranks": [], "latencies": []},
        "noisy_15db": {"hits_top1": 0, "hits_top3": 0, "reciprocal_ranks": [], "latencies": []}
    }

    sr = 16000
    clip_dur = 8.0  # 8 giây chuẩn

    for idx, entry in enumerate(sampled_entries, start=1):
        wav_path = entry["audio_path"]
        target_id = entry["video_id"]
        if not os.path.exists(wav_path):
            continue

        try:
            y, _ = librosa.load(wav_path, sr=sr)
            # Cắt 8s ở đoạn giữa (từ giây 5 đến 13)
            start_sample = int(sr * 5.0)
            end_sample = int(sr * 13.0)
            if len(y) < end_sample:
                y_clean = y[: int(sr * clip_dur)]
            else:
                y_clean = y[start_sample:end_sample]

            # 1. Clean
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_clean:
                sf.write(f_clean.name, y_clean, sr)
                path_clean = f_clean.name

            t0 = time.time()
            res_clean, _, _ = search_by_melody(path_clean, top_k=5, min_duration=5.0)
            lat_clean = time.time() - t0
            os.remove(path_clean)

            conditions["clean"]["latencies"].append(lat_clean)
            rank_clean = 0
            for r_idx, r in enumerate(res_clean, 1):
                if r["video_id"] == target_id:
                    rank_clean = r_idx
                    break
            if rank_clean == 1:
                conditions["clean"]["hits_top1"] += 1
            if 1 <= rank_clean <= 3:
                conditions["clean"]["hits_top3"] += 1
            conditions["clean"]["reciprocal_ranks"].append(1.0 / rank_clean if rank_clean > 0 else 0.0)

            # 2. Pitch Shift +2 Semitones (giả lập người dùng hát lệch tông cao hơn)
            y_shift = librosa.effects.pitch_shift(y_clean, sr=sr, n_steps=2)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_shift:
                sf.write(f_shift.name, y_shift, sr)
                path_shift = f_shift.name

            t0 = time.time()
            res_shift, _, _ = search_by_melody(path_shift, top_k=5, min_duration=5.0)
            lat_shift = time.time() - t0
            os.remove(path_shift)

            conditions["pitch_shift_2"]["latencies"].append(lat_shift)
            rank_shift = 0
            for r_idx, r in enumerate(res_shift, 1):
                if r["video_id"] == target_id:
                    rank_shift = r_idx
                    break
            if rank_shift == 1:
                conditions["pitch_shift_2"]["hits_top1"] += 1
            if 1 <= rank_shift <= 3:
                conditions["pitch_shift_2"]["hits_top3"] += 1
            conditions["pitch_shift_2"]["reciprocal_ranks"].append(1.0 / rank_shift if rank_shift > 0 else 0.0)

            # 3. Noisy (Thêm nhiễu trắng SNR ~15dB - mô phỏng quán cà phê / ồn ào)
            signal_power = np.mean(y_clean ** 2)
            noise_power = signal_power / (10 ** (15 / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), len(y_clean)).astype(np.float32)
            y_noisy = y_clean + noise

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_noisy:
                sf.write(f_noisy.name, y_noisy, sr)
                path_noisy = f_noisy.name

            t0 = time.time()
            res_noisy, _, _ = search_by_melody(path_noisy, top_k=5, min_duration=5.0)
            lat_noisy = time.time() - t0
            os.remove(path_noisy)

            conditions["noisy_15db"]["latencies"].append(lat_noisy)
            rank_noisy = 0
            for r_idx, r in enumerate(res_noisy, 1):
                if r["video_id"] == target_id:
                    rank_noisy = r_idx
                    break
            if rank_noisy == 1:
                conditions["noisy_15db"]["hits_top1"] += 1
            if 1 <= rank_noisy <= 3:
                conditions["noisy_15db"]["hits_top3"] += 1
            conditions["noisy_15db"]["reciprocal_ranks"].append(1.0 / rank_noisy if rank_noisy > 0 else 0.0)

            if idx % 10 == 0 or idx == total_eval:
                print(f"  [{idx}/{total_eval}] Đã test {idx} bài (Clean Top-1: {conditions['clean']['hits_top1']}/{idx})")

        except Exception as e:
            print(f"  ❌ Lỗi khi test bài {target_id}: {e}")

    # Tổng hợp thống kê QBH
    qbh_stats = {}
    for cond_name, d in conditions.items():
        n = max(1, len(d["reciprocal_ranks"]))
        top1_acc = (d["hits_top1"] / n) * 100
        top3_acc = (d["hits_top3"] / n) * 100
        mrr = float(np.mean(d["reciprocal_ranks"]))
        avg_lat = float(np.mean(d["latencies"])) * 1000  # ms
        qbh_stats[cond_name] = {
            "evaluated_samples": n,
            "top1_accuracy": round(top1_acc, 2),
            "top3_accuracy": round(top3_acc, 2),
            "mrr": round(mrr, 4),
            "avg_latency_ms": round(avg_lat, 1)
        }

    return qbh_stats


def evaluate_lyrics_accuracy():
    print("\n" + "=" * 60)
    print("🎤 2. ĐÁNH GIÁ ĐỘ CHÍNH XÁC TÌM KIẾM LỜI BÀI HÁT (LYRICS SEARCH)")
    print("=" * 60)

    wer_path = os.path.join(BASE_DIR, "dataset", "wer_evaluation_results.json")
    if not os.path.exists(wer_path):
        print(f"❌ Không tìm thấy {wer_path}")
        return {}

    with open(wer_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)["per_track_results"]

    sbert_model = SentenceTransformer(SBERT_MODEL, device="cpu")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(str(ID_MAPPING_PATH), "r", encoding="utf-8") as f:
        chunk_mapping = json.load(f)

    total_tracks = len(eval_data)
    print(f"📊 Đang đánh giá Retrieval trên toàn bộ {total_tracks} bài hát...")

    hits_top1 = 0
    hits_top3 = 0
    reciprocal_ranks = []
    latencies = []

    for idx, item in enumerate(eval_data, start=1):
        query_text = item.get("hyp_text_preview", "").strip()
        target_id = item["track_id"]

        if not query_text or len(query_text) < 5:
            # Nếu Whisper không nghe được gì (rỗng hoặc quá ngắn)
            reciprocal_ranks.append(0.0)
            continue

        t0 = time.time()
        q_vec = sbert_model.encode([query_text], normalize_embeddings=True)
        distances, indices = index.search(np.array(q_vec).astype("float32"), 30)
        lat = time.time() - t0
        latencies.append(lat)

        # Lấy danh sách unique video_ids trong kết quả
        seen = set()
        matched_ids = []
        for match_idx in indices[0]:
            mid = chunk_mapping[match_idx]["mongo_id"]
            if mid not in seen:
                seen.add(mid)
                matched_ids.append(mid)

        rank = 0
        for r_idx, mid in enumerate(matched_ids, 1):
            if mid == target_id:
                rank = r_idx
                break

        if rank == 1:
            hits_top1 += 1
        if 1 <= rank <= 3:
            hits_top3 += 1
        reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)

        if idx % 100 == 0 or idx == total_tracks:
            print(f"  [{idx}/{total_tracks}] Top-1 Hits: {hits_top1}/{idx} ({(hits_top1/idx)*100:.1f}%)")

    n = max(1, len(reciprocal_ranks))
    lyrics_stats = {
        "evaluated_samples": n,
        "top1_accuracy": round((hits_top1 / n) * 100, 2),
        "top3_accuracy": round((hits_top3 / n) * 100, 2),
        "mrr": round(float(np.mean(reciprocal_ranks)), 4),
        "avg_latency_ms": round(float(np.mean(latencies)) * 1000, 1)
    }

    return lyrics_stats


def generate_benchmark_report_and_plots(qbh_stats, lyrics_stats):
    print("\n" + "=" * 60)
    print("📈 TỔNG HỢP VÀ VẼ BIỂU ĐỒ SO SÁNH HIỆU NĂNG")
    print("=" * 60)

    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "qbh_engine": qbh_stats,
        "lyrics_engine": lyrics_stats
    }

    with open(BENCHMARK_JSON, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Đã lưu báo cáo chi tiết vào: {BENCHMARK_JSON}")

    # In bảng kết quả
    print("\n" + "-" * 75)
    print(f"{'Phương thức tìm kiếm':<30} | {'Top-1 Acc':<10} | {'Top-3 Acc':<10} | {'MRR':<8} | {'Latency':<10}")
    print("-" * 75)
    print(f"{'🎤 Lời bài hát (Whisper + SBERT)':<30} | {lyrics_stats['top1_accuracy']:>8.1f}% | {lyrics_stats['top3_accuracy']:>8.1f}% | {lyrics_stats['mrr']:>8.4f} | {lyrics_stats['avg_latency_ms']:>6.1f} ms")
    print(f"{'🎵 Giai điệu (QBH - Chuẩn 8s)':<30} | {qbh_stats['clean']['top1_accuracy']:>8.1f}% | {qbh_stats['clean']['top3_accuracy']:>8.1f}% | {qbh_stats['clean']['mrr']:>8.4f} | {qbh_stats['clean']['avg_latency_ms']:>6.1f} ms")
    print(f"{'🎵 Giai điệu (Lệch tông +2 semi)':<30} | {qbh_stats['pitch_shift_2']['top1_accuracy']:>8.1f}% | {qbh_stats['pitch_shift_2']['top3_accuracy']:>8.1f}% | {qbh_stats['pitch_shift_2']['mrr']:>8.4f} | {qbh_stats['pitch_shift_2']['avg_latency_ms']:>6.1f} ms")
    print(f"{'🎵 Giai điệu (Nhiễu SNR 15dB)':<30} | {qbh_stats['noisy_15db']['top1_accuracy']:>8.1f}% | {qbh_stats['noisy_15db']['top3_accuracy']:>8.1f}% | {qbh_stats['noisy_15db']['mrr']:>8.4f} | {qbh_stats['noisy_15db']['avg_latency_ms']:>6.1f} ms")
    print("-" * 75)

    # Vẽ biểu đồ so sánh
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    categories = [
        "Lyrics Search\n(Whisper ASR)",
        "QBH Giai điệu\n(Chuẩn 8s)",
        "QBH Lệch tông\n(+2 Semitones)",
        "QBH Nhiễu ồn\n(SNR 15dB)"
    ]

    top1_vals = [
        lyrics_stats["top1_accuracy"],
        qbh_stats["clean"]["top1_accuracy"],
        qbh_stats["pitch_shift_2"]["top1_accuracy"],
        qbh_stats["noisy_15db"]["top1_accuracy"]
    ]

    top3_vals = [
        lyrics_stats["top3_accuracy"],
        qbh_stats["clean"]["top3_accuracy"],
        qbh_stats["pitch_shift_2"]["top3_accuracy"],
        qbh_stats["noisy_15db"]["top3_accuracy"]
    ]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax1.bar(x - width/2, top1_vals, width, label='Top-1 Accuracy', color='#1DB954', edgecolor='white')
    rects2 = ax1.bar(x + width/2, top3_vals, width, label='Top-3 Accuracy', color='#4ECDC4', edgecolor='white')

    ax1.set_ylabel('Độ chính xác (%)', fontsize=11)
    ax1.set_title('So sánh Độ chính xác (Top-1 & Top-3)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9)
    ax1.set_ylim(0, 110)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Biểu đồ Latency
    lat_vals = [
        lyrics_stats["avg_latency_ms"],
        qbh_stats["clean"]["avg_latency_ms"],
        qbh_stats["pitch_shift_2"]["avg_latency_ms"],
        qbh_stats["noisy_15db"]["avg_latency_ms"]
    ]
    colors_lat = ['#FF6B6B', '#F39C12', '#F1C40F', '#E67E22']
    bars_lat = ax2.bar(categories, lat_vals, color=colors_lat, edgecolor='white', width=0.55)
    ax2.set_ylabel('Thời gian phản hồi (mili-giây)', fontsize=11)
    ax2.set_title('Thời gian xử lý trung bình mỗi truy vấn (Latency)', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    for bar in bars_lat:
        h = bar.get_height()
        ax2.annotate(f'{h:.1f} ms', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(BENCHMARK_CHART, dpi=200)
    plt.close()
    print(f"📊 Đã lưu biểu đồ so sánh vào: {BENCHMARK_CHART}")


if __name__ == "__main__":
    qbh_res = evaluate_qbh_accuracy(num_samples=50)
    lyrics_res = evaluate_lyrics_accuracy()
    generate_benchmark_report_and_plots(qbh_res, lyrics_res)
    print("\n🎉 TOÀN BỘ ĐÁNH GIÁ ACCURACY ĐÃ HOÀN THÀNH!")
