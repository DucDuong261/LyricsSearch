import streamlit as st
import faiss
import json
import numpy as np
import torch
import librosa
import os
import tempfile
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from audio_recorder_streamlit import audio_recorder

from config import (
    get_tracks_collection, PROJECT_ROOT,
    FAISS_INDEX_PATH, ID_MAPPING_PATH,
    WHISPER_MODEL, SBERT_MODEL,
    MIN_AUDIO_DURATION, LYRICS_SEARCH_THRESHOLD
)
from melody_engine import validate_audio_duration, search_by_melody
from hybrid_engine import search_hybrid
from ui_components import (
    save_audio_to_temp, cleanup_temp_file,
    display_hybrid_results, display_qbh_results,
    display_lyrics_results
)

# ==========================================
# CẤU HÌNH TRANG & CSS
# ==========================================
st.set_page_config(page_title="AI Music Matcher", page_icon="🎧", layout="centered", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .app-title { font-size: 2.8rem; font-weight: 900; color: #1DB954; text-align: center; margin-bottom: 5px; }
    .app-subtitle { color: var(--text-color); opacity: 0.7; font-size: 1.1rem; text-align: center; margin-bottom: 30px; }
    .result-card { background-color: var(--secondary-background-color); border-radius: 16px; padding: 24px; border: 1px solid var(--border-color); margin-top: 20px; margin-bottom: 20px; }
    .badge-top1 { background-color: #1DB954; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: bold; }
    .badge-qbh { background-color: #FF6B6B; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: bold; }
    .badge-hybrid { background: linear-gradient(135deg, #1DB954 0%, #3B82F6 100%); color: white; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: bold; }
    .song-title { font-size: 1.8rem; font-weight: 800; margin: 12px 0 4px 0; color: var(--text-color); }
    .song-artist { font-size: 1.1rem; color: var(--text-color); opacity: 0.8; margin: 0 0 16px 0; }
    .lyric-box { background-color: var(--background-color); border-left: 4px solid #1DB954; padding: 12px; border-radius: 4px; font-style: italic; color: var(--text-color); }
    .score-text { color: var(--text-color); opacity: 0.6; font-size: 0.85rem; margin-top: 15px; margin-bottom: 0px; }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. LOAD AI (dùng config constants)
# ==========================================
BASE_DIR = str(PROJECT_ROOT)


@st.cache_resource(show_spinner="Đang đánh thức bộ não AI...")
def load_all_systems():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    whisper_model = pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL,
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        generate_kwargs={"task": "transcribe"}
    )
    # Đặt SBERT trên CPU để tiết kiệm VRAM cho GPU (chỉ tốn vài ms)
    sbert_model = SentenceTransformer(SBERT_MODEL, device="cpu")

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(str(ID_MAPPING_PATH), 'r', encoding='utf-8') as f:
        chunk_mapping = json.load(f)
    tracks_col = get_tracks_collection()
    return whisper_model, sbert_model, index, chunk_mapping, tracks_col


whisper_model, sbert_model, index, chunk_mapping, tracks_col = load_all_systems()


# ==========================================
# 2. LOGIC XỬ LÝ LÕI
# ==========================================
def search_from_text(query_text, top_k=3):
    query_vector = sbert_model.encode([query_text], normalize_embeddings=True)
    fetch_k = min(30, index.ntotal)
    distances, indices = index.search(np.array(query_vector).astype('float32'), fetch_k)
    seen_ids = set()
    seen_songs = set()
    results = []
    for i in range(fetch_k):
        if len(results) >= top_k:
            break
        match_info = chunk_mapping[indices[0][i]]
        mongo_id = match_info["mongo_id"]
        if mongo_id in seen_ids:
            continue
        seen_ids.add(mongo_id)

        track = tracks_col.find_one({"_id": mongo_id})
        if track:
            title = track['metadata']['title']
            artist_str = ", ".join(track['metadata']['artist'])
            song_signature = (title.strip().lower(), artist_str.strip().lower())

            # Tránh trùng lặp cùng 1 bài hát có nhiều video YouTube khác nhau
            if song_signature in seen_songs:
                continue
            seen_songs.add(song_signature)

            raw_path = track.get('snippet', {}).get('path')
            if isinstance(raw_path, str) and raw_path.strip():
                safe_path = raw_path.replace('\\', '/')
                if not os.path.isabs(safe_path):
                    safe_path = os.path.join(BASE_DIR, safe_path)
            else:
                safe_path = ""

            results.append({
                "title": title,
                "artist": artist_str,
                "matched_line": match_info["matched_line"],
                "score": float(distances[0][i]),
                "audio_path": safe_path,
                "full_lyrics": track.get('lyrics', {}).get('full_text', '')
            })
    return results


def process_audio(audio_path):
    is_valid, duration, msg = validate_audio_duration(audio_path, min_duration=MIN_AUDIO_DURATION)
    if not is_valid:
        return [], msg

    audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
    result = whisper_model({"array": audio_array, "sampling_rate": sampling_rate})
    text = result["text"].strip()

    # LỌC NHIỄU: Nếu Whisper nghe ra âm thanh vô nghĩa quá ngắn
    if not text or len(text) < 5:
        return [], "Nhiễu/Không nghe rõ"

    return search_from_text(text), text


# ==========================================
# 3. HELPER: XỬ LÝ AUDIO INPUT (MIC / FILE)
# ==========================================
def _handle_hybrid_input(audio_bytes):
    """Xử lý audio input cho chế độ Hybrid và hiển thị kết quả."""
    with st.spinner("AI đang phân tích song song cả Lời bài hát (Whisper) và Giai điệu (DTW)..."):
        tmp_path = save_audio_to_temp(audio_bytes)
        results, trans, query_feats, msg = search_hybrid(
            tmp_path, whisper_model, sbert_model, index, chunk_mapping,
            tracks_col, top_k=3, min_duration=MIN_AUDIO_DURATION
        )
        display_hybrid_results(results, trans, query_feats, msg)
        cleanup_temp_file(tmp_path)


def _handle_qbh_input(audio_bytes):
    """Xử lý audio input cho chế độ QBH và hiển thị kết quả."""
    with st.spinner("AI đang trích xuất cao độ F0 và dóng hàng giai điệu (DTW)..."):
        tmp_path = save_audio_to_temp(audio_bytes)
        results, query_feats, msg = search_by_melody(
            tmp_path, top_k=3, min_duration=MIN_AUDIO_DURATION
        )
        display_qbh_results(results, query_feats, msg)
        cleanup_temp_file(tmp_path)


def _handle_lyrics_input(audio_bytes):
    """Xử lý audio input cho chế độ Lyrics và hiển thị kết quả."""
    with st.spinner("AI đang bóc tách âm thanh và nhận dạng câu hát..."):
        tmp_path = save_audio_to_temp(audio_bytes)
        results, text = process_audio(tmp_path)
        display_lyrics_results(results, text, threshold=LYRICS_SEARCH_THRESHOLD)
        cleanup_temp_file(tmp_path)


# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================
with st.sidebar:
    st.markdown("## 🎶 Điều hướng")
    st.markdown("---")
    menu = st.radio("", [
        "🌟 Nhận diện Thông minh (Hybrid)",
        "🎤 Chỉ tìm qua Lời hát",
        "🎵 Chỉ tìm qua Giai điệu (QBH)",
        "✍️ Tìm qua Lời gõ tay",
        "📊 Phân tích Tín hiệu",
        "📈 Đánh giá WER/CER"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Đồ án XLAT: Hệ thống Tìm kiếm Âm nhạc Đa phương thức (Hybrid AI).")

st.markdown('<div class="app-title">AI Music Matcher</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Hệ thống tìm kiếm bài hát thông minh: Kết hợp đồng thời Lời hát & Giai điệu ngân nga</div>', unsafe_allow_html=True)

# ==========================================
# CHẾ ĐỘ 0: NHẬN DIỆN THÔNG MINH (HYBRID FUSION)
# ==========================================
if menu == "🌟 Nhận diện Thông minh (Hybrid)":
    st.markdown("<h3 style='text-align: center;'>🌟 Nhận diện Thông minh Toàn diện (Hybrid Match)</h3>", unsafe_allow_html=True)
    st.info("💡 **Trải nghiệm tối giản:** Bạn chỉ cần thu âm hoặc tải file âm thanh (tối thiểu **5.0 giây**). Dù bạn vừa hát vài câu, vừa ngân nga *'la la la'*, hay huýt sáo — Hệ thống sẽ tự động kích hoạt **song song cả 2 động cơ** (ASR Lời hát & DTW Giai điệu) để mang lại kết quả chính xác nhất!")

    tab_mic_h, tab_upload_h = st.tabs(["🎙️ Thu âm trực tiếp", "📁 Tải file Audio lên"])

    with tab_mic_h:
        st.markdown("<h5 style='text-align: center; margin-top: 10px;'>Chạm vào Micro để bắt đầu hát / ngân nga</h5>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            hybrid_bytes = audio_recorder(text="", recording_color="#FF4136", neutral_color="#3B82F6", icon_name="microphone-lines", icon_size="5x", key="mic_hybrid")
        if hybrid_bytes:
            st.markdown("<br>", unsafe_allow_html=True)
            st.audio(hybrid_bytes, format="audio/wav")
            if st.button("🚀 Nhận Diện Thông Minh", type="primary", use_container_width=True, key="btn_mic_hybrid"):
                _handle_hybrid_input(hybrid_bytes)

    with tab_upload_h:
        st.markdown("<h5 style='text-align: center; margin-top: 10px;'>Tải lên file âm thanh (tối thiểu 5.0s)</h5>", unsafe_allow_html=True)
        uploaded_hybrid = st.file_uploader("Định dạng hỗ trợ: WAV, MP3", type=["wav", "mp3"], key="uploader_hybrid")
        if uploaded_hybrid is not None:
            st.audio(uploaded_hybrid, format="audio/wav")
            if st.button("🚀 Phân Tích Thông Minh File", type="primary", use_container_width=True, key="btn_file_hybrid"):
                _handle_hybrid_input(uploaded_hybrid.getvalue())

# ==========================================
# CHẾ ĐỘ 1: CHỈ NHẬN DIỆN QUA LỜI HÁT
# ==========================================
elif menu == "🎤 Chỉ tìm qua Lời hát":
    st.markdown("<h3 style='text-align: center;'>🎤 Nhận diện qua Lời bài hát</h3>", unsafe_allow_html=True)
    st.info("💡 **Hướng dẫn:** Hát hoặc đọc một vài câu lời bài hát mà bạn nhớ (tối thiểu **5.0 giây**). AI sẽ nhận dạng tiếng nói (Whisper ASR) và tìm kiếm ngữ nghĩa lời bài hát.")

    tab_mic, tab_upload = st.tabs(["🎙️ Thu âm qua Micro", "📁 Tải file Audio lên"])

    with tab_mic:
        st.markdown("<h5 style='text-align: center; margin-top: 10px;'>Chạm vào Micro để bắt đầu thu âm</h5>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            audio_bytes = audio_recorder(text="", recording_color="#FF4136", neutral_color="#1DB954", icon_name="microphone", icon_size="5x", key="mic_lyrics")
        if audio_bytes:
            st.markdown("<br>", unsafe_allow_html=True)
            st.audio(audio_bytes, format="audio/wav")
            if st.button("🚀 Xử Lý Bản Thu Lời Hát", type="primary", use_container_width=True, key="btn_mic_lyrics"):
                _handle_lyrics_input(audio_bytes)

    with tab_upload:
        st.markdown("<h5 style='text-align: center; margin-top: 10px;'>Tải lên đoạn ghi âm câu hát (tối thiểu 5.0s)</h5>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Định dạng hỗ trợ: WAV, MP3", type=["wav", "mp3"], key="uploader_lyrics")
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            if st.button("🚀 Phân Tích File Lời Hát", type="primary", use_container_width=True, key="btn_file_lyrics"):
                _handle_lyrics_input(uploaded_file.getvalue())

# ==========================================
# CHẾ ĐỘ 2: NHẬN DIỆN QUA GIAI ĐIỆU (QBH)
# ==========================================
elif menu == "🎵 Chỉ tìm qua Giai điệu (QBH)":
    st.markdown("<h3 style='text-align: center;'>🎵 Query by Humming / Huýt sáo</h3>", unsafe_allow_html=True)
    st.info("💡 **Mẹo:** Ngân nga giai điệu (*'hừm hừm'*, *'la la la'*) hoặc huýt sáo một đoạn điệp khúc mà bạn nhớ từ **5 đến 10 giây**. Hệ thống sử dụng thuật toán **Subsequence DTW** và **Chroma CENS** để dóng hàng đường cong cao độ và tìm bài hát tương ứng.")

    tab_mic_qbh, tab_upload_qbh = st.tabs(["🎙️ Ngân nga qua Micro", "📁 Tải file Giai điệu"])

    with tab_mic_qbh:
        st.markdown("<h5 style='text-align: center; margin-top: 10px;'>Chạm vào Micro để bắt đầu ngân nga / huýt sáo</h5>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            qbh_bytes = audio_recorder(text="", recording_color="#FF4136", neutral_color="#FF6B6B", icon_name="music-note", icon_size="5x", key="mic_qbh")
        if qbh_bytes:
            st.markdown("<br>", unsafe_allow_html=True)
            st.audio(qbh_bytes, format="audio/wav")
            if st.button("🚀 Tìm Bài Hát Qua Giai Điệu", type="primary", use_container_width=True, key="btn_mic_qbh"):
                _handle_qbh_input(qbh_bytes)

    with tab_upload_qbh:
        st.markdown("<h5 style='text-align: center; margin-top: 10px;'>Tải lên file âm thanh ngân nga (WAV/MP3)</h5>", unsafe_allow_html=True)
        uploaded_qbh = st.file_uploader("Định dạng hỗ trợ: WAV, MP3", type=["wav", "mp3"], key="uploader_qbh")
        if uploaded_qbh is not None:
            st.audio(uploaded_qbh, format="audio/wav")
            if st.button("🚀 Phân Tích Giai Điệu File", type="primary", use_container_width=True, key="btn_file_qbh"):
                _handle_qbh_input(uploaded_qbh.getvalue())

# ==========================================
# CHẾ ĐỘ 3: TÌM QUA LỜI NHẠC GÕ TAY
# ==========================================
elif menu == "✍️ Tìm qua Lời gõ tay":
    st.markdown("<h3 style='text-align: center;'>✍️ Bạn nhớ được câu hát nào?</h3>", unsafe_allow_html=True)
    user_query = st.text_input("Gõ trực tiếp vào đây:", placeholder="Ví dụ: Bên em bao lâu...")
    if st.button("🚀 Tìm Bài Hát", type="primary", use_container_width=True):
        if len(user_query.strip()) < 3:
            st.warning("Vui lòng nhập lời bài hát dài hơn một chút!")
        else:
            with st.spinner('Đang rà soát dữ liệu...'):
                results = search_from_text(user_query)
                display_lyrics_results(results, threshold=LYRICS_SEARCH_THRESHOLD)

# ==========================================
# CHẾ ĐỘ 4: PHÂN TÍCH TÍN HIỆU
# ==========================================
elif menu == "📊 Phân tích Tín hiệu":
    st.markdown('<div class="app-title">Phân tích Tín hiệu Âm thanh</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Trực quan hóa các đặc trưng tín hiệu số (Spectrogram, MFCC, Chromagram,...)</div>', unsafe_allow_html=True)

    # Chọn bài hát để phân tích
    all_tracks = list(tracks_col.find({"snippet.path": {"$ne": None}}, {"metadata.title": 1, "metadata.artist": 1, "snippet.path": 1}).limit(50))
    track_options = {f"{t['metadata']['title']} - {', '.join(t['metadata']['artist'])}": t for t in all_tracks}

    selected_name = st.selectbox("🎵 Chọn bài hát để phân tích:", list(track_options.keys()))
    if selected_name and st.button("📊 Phân tích Tín hiệu", type="primary", use_container_width=True):
        selected_track = track_options[selected_name]
        raw_path = selected_track.get('snippet', {}).get('path', '')
        if isinstance(raw_path, str):
            audio_file = raw_path.replace('\\', '/')
            if not os.path.isabs(audio_file):
                audio_file = os.path.join(BASE_DIR, audio_file)
        else:
            audio_file = ''

        if os.path.exists(audio_file):
            with st.spinner("Đang trích xuất đặc trưng tín hiệu..."):
                from audio_analysis import extract_audio_features, plot_full_analysis, get_feature_summary
                features = extract_audio_features(audio_file)

                # Hiển thị thông số kỹ thuật
                summary = get_feature_summary(features)
                st.markdown("#### 📋 Thông số kỹ thuật")
                col_a, col_b = st.columns(2)
                items = list(summary.items())
                for k, v in items[:len(items)//2]:
                    col_a.metric(k.replace('_', ' ').title(), v)
                for k, v in items[len(items)//2:]:
                    col_b.metric(k.replace('_', ' ').title(), v)

                st.divider()

                # Hiển thị bảng phân tích đầy đủ
                st.markdown("#### 📈 Biểu đồ phân tích 7 đặc trưng")
                fig = plot_full_analysis(features, title=selected_name)
                st.pyplot(fig)
                plt.close(fig)

                # Phát audio
                st.audio(audio_file, format="audio/wav")
        else:
            st.error(f"Không tìm thấy file audio: {audio_file}")

    # Upload file riêng để phân tích
    st.divider()
    st.markdown("#### 📁 Hoặc tải lên file audio của bạn")
    uploaded_analysis = st.file_uploader("Tải file WAV/MP3", type=["wav", "mp3"], key="analysis_upload")
    if uploaded_analysis and st.button("📊 Phân tích file tải lên", use_container_width=True):
        with st.spinner("Đang phân tích..."):
            tmp_path = save_audio_to_temp(uploaded_analysis.getvalue())
            from audio_analysis import extract_audio_features, plot_full_analysis, get_feature_summary
            features = extract_audio_features(tmp_path)
            summary = get_feature_summary(features)
            st.markdown("#### 📋 Thông số")
            for k, v in summary.items():
                st.text(f"{k}: {v}")
            fig = plot_full_analysis(features, title=uploaded_analysis.name)
            st.pyplot(fig)
            plt.close(fig)
            cleanup_temp_file(tmp_path)

# ==========================================
# CHẾ ĐỘ 5: ĐÁNH GIÁ WER/CER
# ==========================================
elif menu == "📈 Đánh giá WER/CER":
    st.markdown('<div class="app-title">Đánh giá WER/CER</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Phân tích chất lượng nhận diện giọng nói của Whisper trên toàn bộ dataset</div>', unsafe_allow_html=True)

    results_path = os.path.join(BASE_DIR, "dataset", "wer_evaluation_results.json")
    if not os.path.exists(results_path):
        st.warning("⚠️ Chưa có kết quả đánh giá. Chạy `python evaluation.py` trước.")
    else:
        with open(results_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)

        stats = eval_data["statistics"]
        overall = stats["overall"]
        per_track = eval_data["per_track_results"]

        # Thống kê chung
        st.markdown("### 📊 Thống kê tổng quan")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng bài đánh giá", overall["total_tracks"])
        c2.metric("WER Trung bình", f"{overall['wer_mean']:.1%}")
        c3.metric("WER Trung vị", f"{overall['wer_median']:.1%}")
        c4.metric("CER Trung bình", f"{overall['cer_mean']:.1%}")

        # Phân tích lỗi
        st.markdown("### 🔍 Phân tích loại lỗi (Error Breakdown)")
        total_errors = overall['total_substitutions'] + overall['total_insertions'] + overall['total_deletions']
        if total_errors > 0:
            ec1, ec2, ec3, ec4 = st.columns(4)
            ec1.metric("🔄 Substitutions", f"{overall['total_substitutions']:,}", f"{overall['total_substitutions']/total_errors:.1%}")
            ec2.metric("➕ Insertions", f"{overall['total_insertions']:,}", f"{overall['total_insertions']/total_errors:.1%}")
            ec3.metric("➖ Deletions", f"{overall['total_deletions']:,}", f"{overall['total_deletions']/total_errors:.1%}")
            ec4.metric("✅ Correct Hits", f"{overall['total_hits']:,}")

        # Phân bố WER (biểu đồ)
        st.markdown("### 📈 Phân bố chất lượng WER")
        wer_values = [min(r['wer'], 2.0) for r in per_track]  # Cap at 200% for visualization
        fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
        ax_hist.hist(wer_values, bins=30, color='#1DB954', edgecolor='white', alpha=0.8)
        ax_hist.axvline(x=overall['wer_median'], color='#FF6B6B', linestyle='--', label=f"Median: {overall['wer_median']:.1%}")
        ax_hist.set_xlabel('WER (Word Error Rate)')
        ax_hist.set_ylabel('Số bài hát')
        ax_hist.set_title('Phân bố WER trên toàn bộ dataset')
        ax_hist.legend()
        ax_hist.grid(alpha=0.2)
        st.pyplot(fig_hist)
        plt.close(fig_hist)

        # Phân bố theo nhóm
        dist = stats["wer_distribution"]
        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        categories = ['Xuất sắc\n(<20%)', 'Tốt\n(20-40%)', 'TB\n(40-60%)', 'Kém\n(60-80%)', 'Rất kém\n(>80%)']
        values = [dist['excellent_0_20'], dist['good_20_40'], dist['fair_40_60'], dist['poor_60_80'], dist['bad_80_plus']]
        colors = ['#1DB954', '#4ECDC4', '#FFD93D', '#FF9F43', '#FF6B6B']
        ax_bar.bar(categories, values, color=colors, edgecolor='white')
        for i, v in enumerate(values):
            ax_bar.text(i, v + 1, str(v), ha='center', fontweight='bold')
        ax_bar.set_ylabel('Số bài hát')
        ax_bar.set_title('Phân nhóm chất lượng WER')
        ax_bar.grid(axis='y', alpha=0.2)
        st.pyplot(fig_bar)
        plt.close(fig_bar)

        # Theo ngôn ngữ
        st.markdown("### 🌐 So sánh WER theo ngôn ngữ")
        lang_data = stats["by_language"]
        import pandas as pd
        lang_df = pd.DataFrame([
            {"Ngôn ngữ": lang, "Số bài": d["count"], "WER Trung bình": f"{d['wer_mean']:.1%}", "CER Trung bình": f"{d['cer_mean']:.1%}"}
            for lang, d in lang_data.items()
        ])
        st.dataframe(lang_df, use_container_width=True, hide_index=True)

        # Top best / worst
        st.markdown("### 🏆 Top 10 bài WER tốt nhất")
        best_df = pd.DataFrame(stats["top_10_best"])
        best_df['wer'] = best_df['wer'].apply(lambda x: f"{x:.1%}")
        best_df['cer'] = best_df['cer'].apply(lambda x: f"{x:.1%}")
        st.dataframe(best_df, use_container_width=True, hide_index=True)

        st.markdown("### ⚠️ Top 10 bài WER kém nhất")
        worst_df = pd.DataFrame(stats["top_10_worst"])
        worst_df['wer'] = worst_df['wer'].apply(lambda x: f"{x:.1%}")
        worst_df['cer'] = worst_df['cer'].apply(lambda x: f"{x:.1%}")
        st.dataframe(worst_df, use_container_width=True, hide_index=True)