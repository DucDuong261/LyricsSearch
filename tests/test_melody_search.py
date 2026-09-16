"""
Test suite for melody_engine.py (Query by Humming)
Kiểm tra:
  1. Kiểm tra quy chuẩn thời lượng đầu vào (Min Duration >= 5.0s)
  2. Kiểm tra độ chính xác tìm kiếm giai điệu (Top 1 Accuracy)
  3. Đo lường thời gian đáp ứng (Latency Benchmark)
"""

import os
import time
import tempfile
import soundfile as sf
import numpy as np
import librosa
from melody_engine import (
    validate_audio_duration,
    search_by_melody,
    extract_melody_features,
    DEFAULT_DB_PATH,
    MIN_AUDIO_DURATION
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_SNIPPET = os.path.join(BASE_DIR, "dataset", "snippets", "-01oDwXKSuE.wav")


def test_duration_validation():
    print("\n" + "=" * 50)
    print("🧪 TEST 1: Kiểm tra quy chuẩn thời lượng đầu vào (>= 5.0s)")
    print("=" * 50)

    sr = 16000
    # Tạo file ngắn 3 giây (< 5.0s)
    short_audio = np.random.randn(sr * 3).astype(np.float32) * 0.1
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_short:
        sf.write(tmp_short.name, short_audio, sr)
        short_path = tmp_short.name

    is_valid, dur, msg = validate_audio_duration(short_path, min_duration=5.0)
    print(f"File 3.0s: is_valid={is_valid}, duration={dur:.1f}s")
    print(f"Thông báo: {msg}")
    assert not is_valid, "File < 5s phải bị từ chối!"
    os.remove(short_path)

    # Tạo file đủ 7 giây (>= 5.0s)
    long_audio = np.random.randn(sr * 7).astype(np.float32) * 0.1
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_long:
        sf.write(tmp_long.name, long_audio, sr)
        long_path = tmp_long.name

    is_valid, dur, msg = validate_audio_duration(long_path, min_duration=5.0)
    print(f"File 7.0s: is_valid={is_valid}, duration={dur:.1f}s")
    assert is_valid, "File >= 5s phải được chấp nhận!"
    os.remove(long_path)
    print("✅ TEST 1 PASSED: Kiểm tra độ dài âm thanh hoạt động hoàn hảo!")


def test_melody_search_accuracy():
    print("\n" + "=" * 50)
    print("🧪 TEST 2: Kiểm tra độ chính xác tìm kiếm giai điệu (Top 1 Accuracy)")
    print("=" * 50)

    if not os.path.exists(DEFAULT_DB_PATH):
        print("⚠️ Chưa có melody_database.pkl, bỏ qua test này.")
        return

    # Lấy 1 bài hát thật, cắt 1 đoạn 8 giây (từ giây thứ 5 đến 13) để giả lập người dùng ngân nga đoạn đó
    y, sr = librosa.load(SAMPLE_SNIPPET, sr=16000)
    query_clip = y[sr * 5 : sr * 13]  # 8 giây

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_clip:
        sf.write(tmp_clip.name, query_clip, sr)
        clip_path = tmp_clip.name

    t0 = time.time()
    results, query_feats, msg = search_by_melody(clip_path, top_k=3, min_duration=5.0)
    latency = time.time() - t0

    print(f"Thời gian tìm kiếm: {latency:.3f} giây")
    print(f"Trạng thái: {msg}")
    print(f"Số kết quả tìm thấy: {len(results)}")

    for i, r in enumerate(results, 1):
        print(f"  Top {i}: {r['title']} - {r['artist']}")
        print(f"         Score: {r['score']:.4f} | Vị trí: {r['matched_start_sec']:.1f}s - {r['matched_end_sec']:.1f}s | video_id: {r['video_id']}")

    os.remove(clip_path)

    assert len(results) > 0, "Không tìm thấy kết quả nào!"
    top1 = results[0]
    expected_id = os.path.splitext(os.path.basename(SAMPLE_SNIPPET))[0]
    print(f"Target Video ID: {expected_id} | Top 1 Video ID: {top1['video_id']}")
    assert top1['video_id'] == expected_id, f"Top 1 phải là {expected_id}, nhưng lại là {top1['video_id']}"
    assert top1['score'] > 0.70, f"Điểm tương đồng phải cao (> 0.70), thực tế: {top1['score']:.4f}"
    print("✅ TEST 2 PASSED: Nhận diện chính xác bài hát gốc từ đoạn giai điệu 8s!")


def test_latency_benchmark():
    print("\n" + "=" * 50)
    print("🧪 TEST 3: Benchmark thời gian đáp ứng (Latency)")
    print("=" * 50)

    if not os.path.exists(DEFAULT_DB_PATH):
        return

    y, sr = librosa.load(SAMPLE_SNIPPET, sr=16000)
    clip = y[: sr * 6]  # 6 giây
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, clip, sr)
        clip_path = tmp.name

    latencies = []
    for _ in range(3):
        t0 = time.time()
        search_by_melody(clip_path, top_k=3)
        latencies.append(time.time() - t0)

    os.remove(clip_path)
    avg_lat = sum(latencies) / len(latencies)
    print(f"Thời gian xử lý trung bình: {avg_lat:.3f}s (Lần lượt: {[f'{x:.3f}s' for x in latencies]})")
    assert avg_lat < 2.0, f"Latency quá cao ({avg_lat:.3f}s > 2.0s)"
    print("✅ TEST 3 PASSED: Độ trễ đạt chuẩn thời gian thực (< 2s)!")


if __name__ == "__main__":
    test_duration_validation()
    test_melody_search_accuracy()
    test_latency_benchmark()
    print("\n🎉 TOÀN BỘ CÁC BÀI KIỂM THỬ ĐÃ THÀNH CÔNG RỰC RỠ!")
