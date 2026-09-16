"""
Module A: Phân tích & Trực quan hóa Tín hiệu Âm thanh
========================================================
Trích xuất và vẽ các đặc trưng tín hiệu số từ file âm thanh,
bao gồm: Waveform, Mel-Spectrogram, MFCC, Chromagram,
Spectral Centroid, Zero-Crossing Rate, RMS Energy.

Đây là kiến thức cốt lõi của môn Xử lý Âm thanh & Tiếng nói.
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend cho server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ==========================================
# 1. TRÍCH XUẤT ĐẶC TRƯNG ÂM THANH
# ==========================================
def extract_audio_features(audio_path, sr=16000):
    """
    Trích xuất toàn bộ đặc trưng tín hiệu số từ file audio.

    Parameters:
        audio_path: Đường dẫn file WAV
        sr: Tần số lấy mẫu (Sample Rate), mặc định 16kHz (chuẩn Whisper)

    Returns:
        Dictionary chứa tất cả đặc trưng đã tính toán.

    Lý thuyết:
        - Waveform: Biểu diễn biên độ tín hiệu theo thời gian (Time Domain)
        - Mel-Spectrogram: Biến đổi STFT rồi ánh xạ lên thang Mel
          (mô phỏng cách tai người cảm nhận tần số)
        - MFCC: Mel-Frequency Cepstral Coefficients — 13 hệ số đặc trưng
          biểu diễn hình dạng phổ âm thanh, chuẩn trong nhận diện giọng nói
        - Chromagram: 12 nốt nhạc (C, C#, D, ..., B) theo thời gian
          — hữu ích cho nhận diện giai điệu
        - Spectral Centroid: Trọng tâm phổ tần số — thể hiện độ "sáng"
          của âm thanh (giá trị cao = âm thanh sắc hơn)
        - Zero-Crossing Rate: Tỷ lệ tín hiệu đổi dấu — phân biệt
          voiced (giọng nói) và unvoiced (phụ âm, tiếng ồn)
        - RMS Energy: Năng lượng trung bình bình phương gốc —
          đo "cường độ" âm thanh theo thời gian
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr)
    duration = len(y) / sr

    # 1. Mel-Spectrogram (128 Mel bands, FFT window 2048, hop 512)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 2. MFCC (13 hệ số — chuẩn trong ASR)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)

    # 3. Chromagram (12 nốt nhạc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)

    # 4. Spectral Centroid (trọng tâm phổ)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]

    # 5. Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]

    # 6. RMS Energy
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

    # Time axis for frame-level features
    frames = np.arange(len(spectral_centroid))
    time_frames = librosa.frames_to_time(frames, sr=sr, hop_length=512)
    time_samples = np.linspace(0, duration, len(y))

    return {
        "y": y,
        "sr": sr,
        "duration": duration,
        "mel_spec_db": mel_spec_db,
        "mfccs": mfccs,
        "chroma": chroma,
        "spectral_centroid": spectral_centroid,
        "zcr": zcr,
        "rms": rms,
        "time_frames": time_frames,
        "time_samples": time_samples,
    }


# ==========================================
# 2. TRỰC QUAN HÓA (PLOT)
# ==========================================
def plot_full_analysis(features, title="Audio Analysis", save_path=None):
    """
    Vẽ dashboard phân tích tín hiệu đầy đủ gồm 7 biểu đồ con.
    Trả về đối tượng Figure để có thể nhúng vào Streamlit.
    """
    y = features["y"]
    sr = features["sr"]
    t_frames = features["time_frames"]
    t_samples = features["time_samples"]

    # Thiết lập hình vẽ đẹp
    fig = plt.figure(figsize=(16, 22))
    gs = gridspec.GridSpec(7, 1, hspace=0.45)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    # Color scheme
    cmap_mel = 'magma'
    cmap_mfcc = 'coolwarm'
    cmap_chroma = 'YlGnBu'

    # ---- 1. Waveform ----
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_samples, y, color='#1DB954', linewidth=0.3, alpha=0.8)
    ax1.set_title("① Waveform (Dạng sóng biên độ theo thời gian)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Thời gian (s)")
    ax1.set_ylabel("Biên độ")
    ax1.set_xlim([0, features["duration"]])
    ax1.grid(alpha=0.2)

    # ---- 2. Mel-Spectrogram ----
    ax2 = fig.add_subplot(gs[1])
    img_mel = librosa.display.specshow(
        features["mel_spec_db"], sr=sr, hop_length=512,
        x_axis='time', y_axis='mel', ax=ax2, cmap=cmap_mel
    )
    ax2.set_title("② Mel-Spectrogram (Phổ tần số trên thang Mel)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Thời gian (s)")
    ax2.set_ylabel("Tần số Mel (Hz)")
    fig.colorbar(img_mel, ax=ax2, format='%+2.0f dB', pad=0.01)

    # ---- 3. MFCC ----
    ax3 = fig.add_subplot(gs[2])
    img_mfcc = librosa.display.specshow(
        features["mfccs"], sr=sr, hop_length=512,
        x_axis='time', ax=ax3, cmap=cmap_mfcc
    )
    ax3.set_title("③ MFCC — 13 Hệ số Cepstral (Đặc trưng chuẩn cho ASR)", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Thời gian (s)")
    ax3.set_ylabel("Hệ số MFCC")
    fig.colorbar(img_mfcc, ax=ax3, pad=0.01)

    # ---- 4. Chromagram ----
    ax4 = fig.add_subplot(gs[3])
    img_chroma = librosa.display.specshow(
        features["chroma"], sr=sr, hop_length=512,
        x_axis='time', y_axis='chroma', ax=ax4, cmap=cmap_chroma
    )
    ax4.set_title("④ Chromagram (12 nốt nhạc theo thời gian)", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Thời gian (s)")
    ax4.set_ylabel("Nốt nhạc")
    fig.colorbar(img_chroma, ax=ax4, pad=0.01)

    # ---- 5. Spectral Centroid ----
    ax5 = fig.add_subplot(gs[4])
    ax5.plot(t_frames, features["spectral_centroid"], color='#FF6B6B', linewidth=1.0)
    ax5.fill_between(t_frames, features["spectral_centroid"], alpha=0.15, color='#FF6B6B')
    ax5.set_title("⑤ Spectral Centroid (Trọng tâm phổ — Độ 'sáng' của âm thanh)", fontsize=11, fontweight='bold')
    ax5.set_xlabel("Thời gian (s)")
    ax5.set_ylabel("Tần số (Hz)")
    ax5.set_xlim([0, features["duration"]])
    ax5.grid(alpha=0.2)

    # ---- 6. Zero-Crossing Rate ----
    ax6 = fig.add_subplot(gs[5])
    ax6.plot(t_frames, features["zcr"], color='#4ECDC4', linewidth=1.0)
    ax6.fill_between(t_frames, features["zcr"], alpha=0.15, color='#4ECDC4')
    ax6.set_title("⑥ Zero-Crossing Rate (Tỷ lệ đổi dấu — Phân biệt Voiced/Unvoiced)", fontsize=11, fontweight='bold')
    ax6.set_xlabel("Thời gian (s)")
    ax6.set_ylabel("ZCR")
    ax6.set_xlim([0, features["duration"]])
    ax6.grid(alpha=0.2)

    # ---- 7. RMS Energy ----
    ax7 = fig.add_subplot(gs[6])
    ax7.plot(t_frames, features["rms"], color='#FFD93D', linewidth=1.0)
    ax7.fill_between(t_frames, features["rms"], alpha=0.2, color='#FFD93D')
    ax7.set_title("⑦ RMS Energy (Năng lượng — Cường độ âm thanh theo thời gian)", fontsize=11, fontweight='bold')
    ax7.set_xlabel("Thời gian (s)")
    ax7.set_ylabel("RMS")
    ax7.set_xlim([0, features["duration"]])
    ax7.grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✅ Đã lưu biểu đồ: {save_path}")

    return fig


def get_feature_summary(features):
    """
    Tạo bảng tóm tắt các thông số kỹ thuật của file audio.
    Hữu ích cho báo cáo và hiển thị trên Streamlit.
    """
    y = features["y"]
    sr = features["sr"]

    return {
        "sample_rate": f"{sr:,} Hz",
        "duration": f"{features['duration']:.2f} giây",
        "total_samples": f"{len(y):,}",
        "channels": "Mono",
        "bit_depth": "32-bit float",
        "max_amplitude": f"{np.abs(y).max():.4f}",
        "rms_energy_mean": f"{features['rms'].mean():.4f}",
        "spectral_centroid_mean": f"{features['spectral_centroid'].mean():.0f} Hz",
        "zcr_mean": f"{features['zcr'].mean():.4f}",
        "mfcc_shape": f"{features['mfccs'].shape[0]} coefficients × {features['mfccs'].shape[1]} frames",
        "mel_spec_shape": f"{features['mel_spec_db'].shape[0]} mel bands × {features['mel_spec_db'].shape[1]} frames",
    }


# ==========================================
# 3. CHẠY DEMO PHÂN TÍCH
# ==========================================
if __name__ == "__main__":
    import glob

    SNIPPETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "snippets")
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "analysis_plots")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Lấy 3 file mẫu để demo
    wav_files = sorted(glob.glob(os.path.join(SNIPPETS_DIR, "*.wav")))[:3]

    if not wav_files:
        print("❌ Không tìm thấy file WAV nào trong dataset/snippets/")
    else:
        for wav_path in wav_files:
            filename = os.path.basename(wav_path).replace('.wav', '')
            print(f"\n{'='*50}")
            print(f"📊 Đang phân tích: {filename}")
            print(f"{'='*50}")

            # Trích xuất đặc trưng
            features = extract_audio_features(wav_path)

            # In thông số kỹ thuật
            summary = get_feature_summary(features)
            for key, value in summary.items():
                print(f"  {key:25s}: {value}")

            # Vẽ biểu đồ
            save_path = os.path.join(OUTPUT_DIR, f"{filename}_analysis.png")
            fig = plot_full_analysis(features, title=f"Audio Signal Analysis: {filename}", save_path=save_path)
            plt.close(fig)

    print(f"\n✅ Hoàn tất! Biểu đồ đã được lưu vào: {OUTPUT_DIR}")
