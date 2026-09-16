"""
Module: melody_engine.py
Mục đích: Xử lý tìm kiếm bài hát theo Giai điệu ngân nga / Huýt sáo (Query by Humming - QBH)
Sử dụng:
  - F0 Pitch Contour (YIN) & Relative Semitone Transpose-Invariance
  - Chroma CENS (Chroma Energy Normalized Statistics) chuẩn MIR
  - Subsequence DTW (Dynamic Time Warping)
  - Quy chuẩn thời lượng đầu vào tối thiểu (Min Duration >= 5.0s)
"""

import os
import time
import pickle
import numpy as np
import librosa
import matplotlib.pyplot as plt
from config import (
    get_tracks_collection, PROJECT_ROOT,
    MELODY_DB_PATH, SNIPPETS_DIR,
    MIN_AUDIO_DURATION, SAMPLE_RATE, HOP_LENGTH
)

BASE_DIR = str(PROJECT_ROOT)
DEFAULT_DB_PATH = str(MELODY_DB_PATH)

# Biến cache in-memory để không phải đọc lại file pkl mỗi lần gọi hàm
_CACHED_MELODY_DB = None


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


def extract_melody_features(audio_path, sr=SAMPLE_RATE, hop_length=HOP_LENGTH):
    """
    Trích xuất các đặc trưng giai điệu phục vụ Query by Humming:
    1. Chroma CENS (12, n_frames): Hòa âm bền vững với tempo/âm sắc
    2. Global Chroma (12,): Vector trung bình chuẩn hóa để lọc nhanh
    3. F0 Pitch Contour (n_frames): Tần số cơ bản bằng YIN
    4. Relative Semitones (n_frames): Bán âm chuẩn hóa trung vị (khử lệch tông giọng)
    """
    y, sr = librosa.load(audio_path, sr=sr)
    duration = float(len(y) / sr)

    # 1. Chroma CENS
    cens = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)

    # 2. Global Chroma Vector (L2-normalized)
    g_chroma = np.mean(cens, axis=1)
    norm = np.linalg.norm(g_chroma)
    if norm > 1e-9:
        g_chroma = g_chroma / norm

    # 3. F0 Pitch Contour (YIN)
    # Dải tần số giọng người: 65Hz (C2) - 1000Hz (B5)
    f0 = librosa.yin(y, fmin=65, fmax=1000, sr=sr, hop_length=hop_length)
    voiced = (f0 > 65.0) & (f0 < 990.0)

    # 4. Relative Semitone (Transpose-invariant)
    rel_pitch = np.zeros_like(f0)
    if np.sum(voiced) > 0:
        # Chuyển Hz -> MIDI scale
        midi_notes = 12.0 * np.log2(f0[voiced] / 440.0) + 69.0
        # Trừ đi trung vị để mọi tông giọng (nam/nữ/trầm/cao) khớp với nhau
        rel_pitch[voiced] = midi_notes - np.median(midi_notes)

    times = librosa.times_like(cens, sr=sr, hop_length=hop_length)

    return {
        "cens": cens,
        "global_chroma": g_chroma,
        "f0": f0,
        "rel_pitch": rel_pitch,
        "voiced": voiced,
        "times": times,
        "duration": duration,
        "n_frames": cens.shape[1]
    }


def build_melody_database(output_path=DEFAULT_DB_PATH):
    """
    Quét toàn bộ bài hát trong dataset/snippets và MongoDB để tạo
    cơ sở dữ liệu giai điệu đã tính toán sẵn (Precomputed Melody DB).
    """
    print("=" * 60)
    print("🚀 BẮT ĐẦU XÂY DỰNG CƠ SỞ DỮ LIỆU GIAI ĐIỆU (MELODY DATABASE)")
    print("=" * 60)

    # Kết nối MongoDB để lấy thông tin metadata (Singleton từ config.py)
    tracks_col = get_tracks_collection()

    # Đọc danh sách file wav trong snippets
    snippets_dir = str(SNIPPETS_DIR)
    wav_files = [f for f in os.listdir(snippets_dir) if f.endswith(".wav")]
    total_files = len(wav_files)
    print(f"📁 Tìm thấy {total_files} file snippet trong thư mục {snippets_dir}")

    db_entries = []
    t_start = time.time()

    for idx, filename in enumerate(wav_files, start=1):
        video_id = os.path.splitext(filename)[0]
        full_path = os.path.join(snippets_dir, filename)

        # Lấy metadata từ MongoDB
        track = tracks_col.find_one({"_id": video_id})
        if track:
            title = track.get("metadata", {}).get("title", video_id)
            artist_list = track.get("metadata", {}).get("artist", ["Unknown"])
            artist = ", ".join(artist_list) if isinstance(artist_list, list) else str(artist_list)
        else:
            title = video_id
            artist = "Unknown Artist"

        try:
            feats = extract_melody_features(full_path)
            db_entries.append({
                "video_id": video_id,
                "title": title,
                "artist": artist,
                "audio_path": full_path,
                "duration": feats["duration"],
                "global_chroma": feats["global_chroma"],
                "cens": feats["cens"].astype(np.float32),
                "f0": feats["f0"].astype(np.float32),
                "rel_pitch": feats["rel_pitch"].astype(np.float32),
                "voiced": feats["voiced"],
                "times": feats["times"].astype(np.float32)
            })
            if idx % 50 == 0 or idx == total_files:
                elapsed = time.time() - t_start
                print(f"  [{idx}/{total_files}] Đã trích xuất {idx} bài ({elapsed:.1f}s, {(idx/total_files)*100:.1f}%)")
        except Exception as e:
            print(f"  ❌ Lỗi khi xử lý {filename}: {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(db_entries, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_time = time.time() - t_start
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("=" * 60)
    print(f"✅ HOÀN TẤT! Đã lưu {len(db_entries)} bài vào: {output_path}")
    print(f"📦 Dung lượng DB: {size_mb:.2f} MB | Thời gian thực thi: {total_time:.2f}s")
    print("=" * 60)
    return db_entries


def load_melody_database(db_path=DEFAULT_DB_PATH):
    """
    Nạp database giai điệu vào bộ nhớ RAM (Singleton Cache).
    """
    global _CACHED_MELODY_DB
    if _CACHED_MELODY_DB is not None:
        return _CACHED_MELODY_DB

    if not os.path.exists(db_path):
        print(f"⚠️ Chưa tìm thấy {db_path}. Đang tiến hành tạo mới...")
        _CACHED_MELODY_DB = build_melody_database(db_path)
    else:
        with open(db_path, "rb") as f:
            _CACHED_MELODY_DB = pickle.load(f)

    return _CACHED_MELODY_DB


def search_by_melody(query_audio_path, top_k=3, min_duration=MIN_AUDIO_DURATION, db_path=DEFAULT_DB_PATH):
    """
    Tìm kiếm bài hát theo giai điệu ngân nga / huýt sáo:
    - Bước 0: Validate thời lượng >= 5.0s
    - Bước 1: Trích xuất đặc trưng query
    - Bước 2: Lọc nhanh Top 25 ứng viên bằng Cosine Similarity trên Global Chroma (< 5ms)
    - Bước 3: So khớp chi tiết bằng Subsequence DTW trên Chroma CENS
    - Bước 4: Chấm điểm tương đồng (Similarity Score 0.0 - 1.0) và định vị thời gian
    """
    # 0. Kiểm tra ngưỡng độ dài
    is_valid, duration, msg = validate_audio_duration(query_audio_path, min_duration)
    if not is_valid:
        return [], None, msg

    # 1. Trích xuất đặc trưng query
    query_feats = extract_melody_features(query_audio_path)
    q_cens = query_feats["cens"]
    q_global = query_feats["global_chroma"]
    n_query_frames = q_cens.shape[1]

    # Nạp cơ sở dữ liệu
    db = load_melody_database(db_path)
    if not db:
        return [], None, "Cơ sở dữ liệu giai điệu rỗng."

    # 2. Lọc nhanh (Fast Screening) qua Transpose-Invariant Cosine Similarity
    # Thử 12 bước dịch chuyển bán âm (12 circular chromatic shifts) để tự động bắt đúng tông giọng
    all_globals = np.array([entry["global_chroma"] for entry in db])  # shape: (N, 12)
    q_shifts = np.array([np.roll(q_global, -k) for k in range(12)])  # shape: (12, 12)
    all_shift_sims = np.dot(all_globals, q_shifts.T)  # shape: (N, 12)
    cosine_sims = np.max(all_shift_sims, axis=1)  # shape: (N,)
    best_shifts = np.argmax(all_shift_sims, axis=1)  # shape: (N,)

    # Chọn Top 25 bài có hòa âm tương đồng nhất (đã tối ưu theo tông giọng)
    num_candidates = min(25, len(db))
    top_candidate_indices = np.argsort(cosine_sims)[::-1][:num_candidates]

    results = []

    # 3. So khớp Subsequence DTW
    for idx in top_candidate_indices:
        entry = db[idx]
        ref_cens = entry["cens"]
        k_shift = int(best_shifts[idx])

        # Dóng hàng tông giọng (circular roll) trước khi chạy DTW
        q_cens_aligned = np.roll(q_cens, -k_shift, axis=0) if k_shift != 0 else q_cens

        # Subsequence DTW: tìm đoạn con trong ref_cens khớp nhất với q_cens_aligned
        D, wp = librosa.sequence.dtw(X=q_cens_aligned, Y=ref_cens, metric='cosine', subseq=True)

        # Chi phí dóng hàng nhỏ nhất ở hàng cuối cùng
        last_row = D[-1, :]
        min_cost = float(np.min(last_row))
        best_end_frame = int(np.argmin(last_row))

        # Tìm frame bắt đầu qua warping path
        best_start_frame = int(wp[-1][1])

        # Chi phí chuẩn hóa trung bình mỗi frame
        norm_cost = min_cost / max(1, n_query_frames)

        # Chuyển đổi cost sang điểm số tương đồng (0.0 đến 1.0)
        similarity = float(np.clip(1.0 - (norm_cost / 0.7), 0.0, 1.0))

        # Tính thời gian (giây) trong bài gốc
        start_sec = best_start_frame * HOP_LENGTH / SAMPLE_RATE
        end_sec = best_end_frame * HOP_LENGTH / SAMPLE_RATE

        results.append({
            "video_id": entry["video_id"],
            "title": entry["title"],
            "artist": entry["artist"],
            "audio_path": entry["audio_path"],
            "score": similarity,
            "dtw_cost": norm_cost,
            "matched_start_sec": start_sec,
            "matched_end_sec": end_sec,
            "matched_start_frame": best_start_frame,
            "matched_end_frame": best_end_frame,
            "ref_entry": entry
        })

    # Sắp xếp kết quả theo điểm tương đồng cao nhất
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k], query_feats, "Thành công"


def plot_melody_alignment(query_feats, matched_result):
    """
    Vẽ biểu đồ dóng hàng đường cong giai điệu (Pitch Alignment)
    giữa tiếng ngân nga của người dùng và bài hát gốc tìm thấy.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    q_rel = query_feats["rel_pitch"]
    q_voiced = query_feats["voiced"]
    q_times = query_feats["times"]

    # 1. Đường cong cao độ người dùng ngân nga
    q_pitch_plot = np.where(q_voiced, q_rel, np.nan)
    ax1.plot(q_times, q_pitch_plot, color="#FF6B6B", lw=2.5, label="Giọng người dùng ngân nga (Pitch)")
    ax1.scatter(q_times[q_voiced], q_pitch_plot[q_voiced], color="#EE5253", s=15, alpha=0.7)
    ax1.set_ylabel("Cao độ tương đối (Semitone)")
    ax1.set_title(f"🎵 Đường cong giai điệu người dùng ({query_feats['duration']:.1f}s)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right")

    # 2. Đoạn giai điệu bài hát gốc tương ứng
    ref_entry = matched_result["ref_entry"]
    ref_rel = ref_entry["rel_pitch"]
    ref_voiced = ref_entry["voiced"]
    ref_times = ref_entry["times"]

    start_f = matched_result["matched_start_frame"]
    end_f = matched_result["matched_end_frame"]

    # Cắt đoạn bài gốc tương ứng
    slice_start = max(0, start_f)
    slice_end = min(len(ref_rel), end_f + 1)

    sub_ref_pitch = np.where(ref_voiced[slice_start:slice_end], ref_rel[slice_start:slice_end], np.nan)
    sub_ref_times = ref_times[slice_start:slice_end] - ref_times[slice_start]  # Reset về 0s để so sánh

    ax2.plot(sub_ref_times, sub_ref_pitch, color="#1DB954", lw=2.5, label=f"Bài hát gốc: {matched_result['title']}")
    ax2.set_xlabel("Thời gian dóng hàng (giây)")
    ax2.set_ylabel("Cao độ tương đối (Semitone)")
    score_disp = matched_result.get('score', matched_result.get('final_score', 0.0))
    ax2.set_title(
        f"🎧 Đoạn giai điệu tương ứng trong bài gốc [{matched_result['matched_start_sec']:.1f}s - {matched_result['matched_end_sec']:.1f}s] (Độ tương đồng: {score_disp*100:.1f}%)",
        fontsize=11,
        fontweight="bold"
    )
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Melody Search Engine (QBH)")
    parser.add_argument("--build-db", action="store_true", help="Xây dựng lại database giai điệu")
    parser.add_argument("--test-file", type=str, default=None, help="File audio test tìm kiếm")
    args = parser.parse_args()

    if args.build_db:
        build_melody_database()

    if args.test_file:
        print(f"\n🔍 Đang tìm kiếm cho file: {args.test_file}")
        results, feats, msg = search_by_melody(args.test_file, top_k=3)
        print(f"Trạng thái: {msg}")
        for idx, r in enumerate(results, 1):
            print(f"Top {idx}: {r['title']} - {r['artist']} | Điểm: {r['score']:.4f} | Đoạn: {r['matched_start_sec']:.1f}s - {r['matched_end_sec']:.1f}s")
