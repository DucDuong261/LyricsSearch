"""
Module C: Tiền xử lý & Nâng cao chất lượng Âm thanh
=====================================================
Áp dụng các kỹ thuật xử lý tín hiệu để cải thiện chất lượng audio
trước khi đưa vào mô hình Whisper nhận diện.

Bao gồm:
  1. Noise Reduction (Khử nhiễu bằng Spectral Gating)
  2. Audio Normalization (Chuẩn hóa biên độ)
  3. Voice Activity Detection (Phát hiện đoạn có giọng nói)
  4. So sánh WER trước/sau tiền xử lý
"""

import os
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr


# ==========================================
# 1. KHỬ NHIỄU (NOISE REDUCTION)
# ==========================================
def reduce_noise(y, sr=16000, prop_decrease=0.8):
    """
    Khử nhiễu nền bằng phương pháp Spectral Gating.

    Nguyên lý hoạt động:
        1. Ước lượng phổ nhiễu (noise profile) từ toàn bộ tín hiệu
        2. Tính ngưỡng phổ (spectral threshold) từ noise profile
        3. Áp dụng "cổng phổ" (spectral gate): các thành phần phổ
           có biên độ dưới ngưỡng sẽ bị giảm biên độ (attenuate)
        4. Biến đổi ISTFT để tái tạo tín hiệu sạch

    Parameters:
        y: Mảng tín hiệu âm thanh
        sr: Tần số lấy mẫu
        prop_decrease: Mức độ giảm nhiễu (0.0 = không giảm, 1.0 = giảm hoàn toàn)

    Returns:
        Tín hiệu đã khử nhiễu (numpy array)
    """
    y_denoised = nr.reduce_noise(
        y=y,
        sr=sr,
        prop_decrease=prop_decrease,
        stationary=False,  # Nhiễu không dừng (non-stationary) phù hợp hơn với nhạc
        n_fft=2048,
        hop_length=512
    )
    return y_denoised


# ==========================================
# 2. CHUẨN HÓA BIÊN ĐỘ (NORMALIZATION)
# ==========================================
def normalize_audio(y, target_rms=0.1):
    """
    Chuẩn hóa biên độ tín hiệu âm thanh.

    Hai bước:
        1. RMS Normalization: Điều chỉnh năng lượng trung bình về mức target_rms
        2. Peak Clipping Prevention: Đảm bảo biên độ không vượt quá [-1.0, 1.0]

    Parameters:
        y: Tín hiệu đầu vào
        target_rms: Năng lượng RMS mong muốn (mặc định 0.1)

    Returns:
        Tín hiệu đã chuẩn hóa
    """
    # Tính RMS hiện tại
    current_rms = np.sqrt(np.mean(y ** 2))
    if current_rms == 0:
        return y

    # Scale về target RMS
    scaling_factor = target_rms / current_rms
    y_normalized = y * scaling_factor

    # Clip để tránh clipping distortion
    y_normalized = np.clip(y_normalized, -1.0, 1.0)

    return y_normalized


# ==========================================
# 3. VOICE ACTIVITY DETECTION (VAD)
# ==========================================
def detect_voice_segments(y, sr=16000, energy_threshold=0.01, min_duration=0.5):
    """
    Phát hiện các đoạn có giọng nói/hát trong tín hiệu âm thanh.

    Nguyên lý:
        1. Tính RMS Energy theo từng frame
        2. Đánh dấu các frame có năng lượng vượt ngưỡng (threshold)
        3. Gộp các frame liên tiếp thành các đoạn (segments)
        4. Loại bỏ các đoạn quá ngắn (< min_duration)

    Parameters:
        y: Tín hiệu đầu vào
        sr: Tần số lấy mẫu
        energy_threshold: Ngưỡng năng lượng tối thiểu để coi là "có tiếng"
        min_duration: Thời lượng tối thiểu của một đoạn (giây)

    Returns:
        List of tuples (start_sec, end_sec) cho các đoạn có tiếng
    """
    # Tính RMS Energy
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Xác định ngưỡng tự động (adaptive threshold) nếu threshold quá thấp
    adaptive_threshold = max(energy_threshold, np.mean(rms) * 0.5)

    # Đánh dấu các frame vượt ngưỡng
    is_voice = rms > adaptive_threshold

    # Gộp thành các đoạn liên tục
    segments = []
    in_segment = False
    start_frame = 0

    for i, active in enumerate(is_voice):
        if active and not in_segment:
            start_frame = i
            in_segment = True
        elif not active and in_segment:
            end_frame = i
            start_sec = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
            end_sec = librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length)
            if end_sec - start_sec >= min_duration:
                segments.append((round(start_sec, 2), round(end_sec, 2)))
            in_segment = False

    # Xử lý segment cuối cùng nếu tín hiệu kết thúc khi đang trong segment
    if in_segment:
        end_sec = librosa.frames_to_time(len(is_voice), sr=sr, hop_length=hop_length)
        start_sec = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
        if end_sec - start_sec >= min_duration:
            segments.append((round(start_sec, 2), round(end_sec, 2)))

    return segments


def extract_voice_only(y, sr=16000, segments=None):
    """
    Trích xuất chỉ phần có giọng nói/hát từ tín hiệu, loại bỏ đoạn im lặng.
    """
    if segments is None:
        segments = detect_voice_segments(y, sr)

    if not segments:
        return y  # Trả về nguyên bản nếu không phát hiện được

    voice_parts = []
    for start_sec, end_sec in segments:
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        voice_parts.append(y[start_sample:end_sample])

    return np.concatenate(voice_parts)


# ==========================================
# 4. PIPELINE TIỀN XỬ LÝ ĐẦY ĐỦ
# ==========================================
def preprocess_audio(audio_path, sr=16000, save_path=None):
    """
    Pipeline tiền xử lý hoàn chỉnh:
        1. Load audio
        2. Khử nhiễu (Spectral Gating)
        3. Chuẩn hóa biên độ (RMS Normalization)
        4. (Tùy chọn) Lưu file đã xử lý

    Returns:
        Tuple (y_processed, sr, processing_info)
    """
    # Load
    y, sr = librosa.load(audio_path, sr=sr)
    original_rms = float(np.sqrt(np.mean(y ** 2)))
    original_max = float(np.abs(y).max())

    # Bước 1: Khử nhiễu
    y_denoised = reduce_noise(y, sr)
    denoised_rms = float(np.sqrt(np.mean(y_denoised ** 2)))

    # Bước 2: Chuẩn hóa
    y_normalized = normalize_audio(y_denoised, target_rms=0.1)
    final_rms = float(np.sqrt(np.mean(y_normalized ** 2)))

    # Bước 3: VAD
    voice_segments = detect_voice_segments(y_normalized, sr)

    # Tính tỷ lệ tiếng nói
    total_voice_duration = sum(end - start for start, end in voice_segments)
    total_duration = len(y_normalized) / sr
    voice_ratio = total_voice_duration / total_duration if total_duration > 0 else 0

    processing_info = {
        "original_rms": round(original_rms, 4),
        "original_max_amplitude": round(original_max, 4),
        "denoised_rms": round(denoised_rms, 4),
        "final_rms": round(final_rms, 4),
        "noise_reduction_ratio": round(1 - denoised_rms / original_rms, 4) if original_rms > 0 else 0,
        "voice_segments": voice_segments,
        "voice_ratio": round(voice_ratio, 4),
        "total_voice_duration": round(total_voice_duration, 2),
        "total_duration": round(total_duration, 2),
    }

    # Lưu file nếu cần
    if save_path:
        sf.write(save_path, y_normalized, sr)

    return y_normalized, sr, processing_info


# ==========================================
# 5. SO SÁNH WER TRƯỚC/SAU TIỀN XỬ LÝ
# ==========================================
def compare_wer_before_after(tracks_sample, whisper_pipeline, sr=16000):
    """
    Chạy Whisper trên audio GỐC vs audio ĐÃ TIỀN XỬ LÝ,
    so sánh WER để chứng minh hiệu quả của Module C.

    Parameters:
        tracks_sample: List of track documents từ MongoDB
        whisper_pipeline: transformers.pipeline đã load sẵn
        sr: Sample rate

    Returns:
        List of comparison results
    """
    from jiwer import wer as compute_wer
    from evaluation import normalize_text, extract_lyrics_at_timerange

    results = []

    for track in tracks_sample:
        title = track["metadata"]["title"]
        audio_path = track.get("snippet", {}).get("path", "")
        if isinstance(audio_path, str):
            audio_path = audio_path.replace('\\', '/')

        synced_lrc = track.get("lyrics", {}).get("synced_lrc")
        full_text = track.get("lyrics", {}).get("full_text", "")
        start_time = track.get("snippet", {}).get("start_time", 60)
        end_time = track.get("snippet", {}).get("end_time", 90)

        ground_truth = extract_lyrics_at_timerange(synced_lrc, start_time, end_time)
        if not ground_truth:
            ground_truth = full_text

        ref = normalize_text(ground_truth)
        if not ref or not os.path.exists(audio_path):
            continue

        print(f"  Đang xử lý: {title[:50]}...", end="", flush=True)

        try:
            # --- Audio gốc ---
            y_raw, _ = librosa.load(audio_path, sr=sr)
            result_raw = whisper_pipeline({"array": y_raw, "sampling_rate": sr})
            hyp_raw = normalize_text(result_raw["text"])

            # --- Audio đã tiền xử lý ---
            y_processed, _, info = preprocess_audio(audio_path, sr=sr)
            result_processed = whisper_pipeline({"array": y_processed, "sampling_rate": sr})
            hyp_processed = normalize_text(result_processed["text"])

            # --- Tính WER ---
            wer_raw = compute_wer(ref, hyp_raw) if hyp_raw else 999.0
            wer_processed = compute_wer(ref, hyp_processed) if hyp_processed else 999.0
            improvement = wer_raw - wer_processed

            results.append({
                "title": title,
                "wer_raw": round(wer_raw, 4),
                "wer_processed": round(wer_processed, 4),
                "improvement": round(improvement, 4),
                "raw_text": hyp_raw[:80],
                "processed_text": hyp_processed[:80],
                "voice_ratio": info["voice_ratio"],
            })

            symbol = "✅" if improvement > 0 else "⚠️"
            print(f" {symbol} WER: {wer_raw:.1%} → {wer_processed:.1%} (Δ{improvement:+.1%})")

        except Exception as e:
            print(f" ❌ Lỗi: {e}")

    return results


# ==========================================
# 6. CHẠY DEMO
# ==========================================
if __name__ == "__main__":
    import glob

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SNIPPETS_DIR = os.path.join(BASE_DIR, "dataset", "snippets")
    OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "preprocessed_samples")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wav_files = sorted(glob.glob(os.path.join(SNIPPETS_DIR, "*.wav")))[:3]

    print("=" * 60)
    print("  MODULE C: TIỀN XỬ LÝ ÂM THANH")
    print("=" * 60)

    for wav_path in wav_files:
        filename = os.path.basename(wav_path)
        print(f"\n📁 File: {filename}")

        save_path = os.path.join(OUTPUT_DIR, f"processed_{filename}")
        y_processed, sr, info = preprocess_audio(wav_path, save_path=save_path)

        print(f"  Original RMS:       {info['original_rms']}")
        print(f"  After Denoise RMS:  {info['denoised_rms']}")
        print(f"  Final RMS:          {info['final_rms']}")
        print(f"  Noise Reduction:    {info['noise_reduction_ratio']:.1%}")
        print(f"  Voice Segments:     {len(info['voice_segments'])} đoạn")
        print(f"  Voice Ratio:        {info['voice_ratio']:.1%}")
        print(f"  ✅ Saved: {save_path}")

    print(f"\n✅ Hoàn tất!")
