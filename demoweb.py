import streamlit as st
import faiss
import json
import numpy as np
import torch
import librosa
import os
import tempfile
from textwrap import dedent
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from audio_recorder_streamlit import audio_recorder

from config import (
    get_tracks_collection, PROJECT_ROOT,
    FAISS_INDEX_PATH, ID_MAPPING_PATH,
    WHISPER_MODEL, WHISPER_LANGUAGE, SBERT_MODEL,
    MIN_AUDIO_DURATION, LYRICS_SEARCH_THRESHOLD
)
from audio_encoder import validate_audio_duration
from hybrid_engine import search_hybrid
from ui_components import (
    save_audio_to_temp, cleanup_temp_file,
    display_hybrid_results,
    display_lyrics_results
)

# ==========================================
# CẤU HÌNH TRANG & CSS LUXURY DARK THEME
# ==========================================
st.set_page_config(
    page_title="AI Music Matcher • Dual-Modality",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
    /* Reset & Typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Background Glow */
    .stApp {
        background: 
            radial-gradient(circle at 50% 0%, rgba(16, 185, 129, 0.08) 0%, transparent 45%),
            radial-gradient(circle at 90% 15%, rgba(99, 102, 241, 0.07) 0%, transparent 40%),
            radial-gradient(circle at 10% 30%, rgba(6, 182, 212, 0.06) 0%, transparent 35%),
            #080C14 !important;
        color: #F1F5F9;
    }
    
    /* Top Header / Hero */
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #10B981 0%, #06B6D4 50%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 6px;
        line-height: 1.15;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.15rem;
        font-weight: 400;
        text-align: center;
        max-width: 800px;
        margin: 0 auto 24px auto;
        line-height: 1.6;
    }
    
    /* Badges Bar */
    .badge-bar {
        display: flex;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 28px;
    }
    
    .badge-item {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 9999px;
        padding: 5px 14px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #CBD5E1;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Card Luxury Style */
    .glass-card {
        background: rgba(18, 24, 38, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(16, 185, 129, 0.3);
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.6), 0 0 20px rgba(16, 185, 129, 0.1);
    }
    
    /* Sample Quick-Test Card */
    .sample-card {
        background: rgba(20, 28, 45, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 18px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        transition: all 0.25s ease;
    }
    .sample-card:hover {
        transform: translateY(-3px);
        border-color: rgba(6, 182, 212, 0.4);
        box-shadow: 0 12px 30px rgba(0,0,0,0.45), 0 0 15px rgba(6, 182, 212, 0.15);
    }
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #94A3B8;
        border: none !important;
        transition: all 0.25s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.22) 0%, rgba(6, 182, 212, 0.15) 100%) !important;
        color: #10B981 !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
    }
    
    /* Custom Button */
    div.stButton > button:first-child {
        border-radius: 12px;
        font-weight: 700;
        letter-spacing: 0.02em;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        padding: 10px 24px;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.35);
    }
    
    /* Custom Input fields */
    .stTextInput > div > div > input {
        border-radius: 12px;
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #F8FAFC !important;
        padding: 12px 18px;
        font-size: 1.05rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #10B981 !important;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.3) !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B0F19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    /* Footer Clean */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. LOAD AI (dùng config constants)
# ==========================================
BASE_DIR = str(PROJECT_ROOT)


@st.cache_resource(show_spinner="Đang đánh thức bộ não AI (Whisper ASR, SBERT, CLAP)...")
def load_all_systems():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    whisper_model = pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL,
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        generate_kwargs={"task": "transcribe", "language": WHISPER_LANGUAGE}
    )
    # Đặt SBERT trên CPU để tiết kiệm VRAM cho GPU
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
    fetch_k = min(40, index.ntotal)
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
            artist_str = ", ".join(track['metadata']['artist']) if isinstance(track['metadata']['artist'], list) else str(track['metadata']['artist'])
            song_signature = (title.strip().lower(), artist_str.strip().lower())

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

    if not text or len(text) < 5:
        return [], "Nhiễu/Không nghe rõ"

    return search_from_text(text), text


# ==========================================
# 3. HELPER: XỬ LÝ AUDIO INPUT (MIC / FILE)
# ==========================================
def _handle_hybrid_input(audio_bytes):
    """Xử lý audio input cho chế độ Hybrid và hiển thị kết quả."""
    with st.spinner("⚡ AI đang phân tích song song: Ngữ âm tiếng Việt (Whisper) & Âm thanh (CLAP Audio Encoder)..."):
        tmp_path = save_audio_to_temp(audio_bytes)
        results, trans, msg = search_hybrid(
            tmp_path, whisper_model, sbert_model, index, chunk_mapping,
            tracks_col, top_k=3, min_duration=MIN_AUDIO_DURATION
        )
        display_hybrid_results(results, trans, msg)
        cleanup_temp_file(tmp_path)


def _handle_lyrics_input(audio_bytes):
    """Xử lý audio input cho chế độ Lyrics và hiển thị kết quả."""
    with st.spinner("🎙️ AI đang bóc tách âm thanh và nhận dạng câu hát tiếng Việt..."):
        tmp_path = save_audio_to_temp(audio_bytes)
        results, text = process_audio(tmp_path)
        display_lyrics_results(results, text, threshold=LYRICS_SEARCH_THRESHOLD)
        cleanup_temp_file(tmp_path)


# ==========================================
# 4. SIDEBAR CHUYÊN NGHIỆP
# ==========================================
with st.sidebar:
    st.html(dedent("""
    <div style="text-align: center; padding: 10px 0 20px 0; font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;">
        <span style="font-size: 2.8rem;">🎧</span>
        <h3 style="margin: 6px 0 2px 0; color: #10B981; font-weight: 800; letter-spacing: -0.02em;">MUSIC MATCHER</h3>
        <span style="font-size: 0.8rem; color: #64748B; letter-spacing: 0.08em; text-transform: uppercase;">Dual-Modality AI</span>
    </div>
    """).strip())
    
    st.markdown("##### 🧭 CHẾ ĐỘ NHẬN DIỆN")
    menu = st.radio("", [
        "🌟 Nhận diện Thông minh (Hybrid)",
        "🎤 Chỉ tìm qua Lời hát (ASR)",
        "✍️ Tìm qua Lời gõ tay",
        "📥 Thu thập Nhạc Mới (AI Agent)",
        "📊 Phân tích Tín hiệu Âm thanh",
        "📈 Đánh giá WER/CER"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("##### 📊 HỆ THỐNG & DỮ LIỆU")
    
    try:
        total_tracks_count = tracks_col.count_documents({})
    except Exception:
        total_tracks_count = 501

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Kho Bài Hát", f"{total_tracks_count:,}", help="Tổng số bài hát tiếng Việt trong MongoDB")
        st.metric("Vector Lời", "14,982", help="Số lượng vector câu hát trong FAISS")
    with col_s2:
        st.metric("Vector Audio", "396+", help="Số lượng vector audio 512d trong Qdrant")
        st.metric("Ngôn Ngữ", "Tiếng Việt", help="100% chuẩn hoá Tiếng Việt")
        
    st.html(dedent("""
    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 12px; margin-top: 14px; font-size: 0.8rem; color: #94A3B8; font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;">
        <div><b>ASR:</b> OpenAI Whisper (vi)</div>
        <div style="margin-top: 4px;"><b>SBERT:</b> MiniLM-L12-v2 (FAISS 384d)</div>
        <div style="margin-top: 4px;"><b>Audio:</b> CLAP HTSAT-tiny (Qdrant 512d)</div>
        <div style="margin-top: 4px;"><b>Tăng tốc:</b> CUDA GPU Accelerated</div>
    </div>
    """).strip())


# ==========================================
# 5. HERO HEADER & PHẦN THỬ NGHIỆM NHANH
# ==========================================
st.markdown('<div class="hero-title">AI Music Matcher</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Hệ thống tìm kiếm bài hát thông minh đa phương thức: Kết hợp đồng thời <b>Ngữ nghĩa Lời hát (SBERT)</b> '
    'và <b>Đặc trưng Âm thanh (CLAP Audio Encoder)</b> để nhận diện tức thì.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(dedent(f"""
<div class="badge-bar">
    <div class="badge-item"><span style="color: #10B981;">●</span> {total_tracks_count} Bài Hát V-Pop</div>
    <div class="badge-item"><span style="color: #06B6D4;">●</span> Dual-Modality Fusion</div>
    <div class="badge-item"><span style="color: #818CF8;">●</span> OpenAI Whisper ASR</div>
    <div class="badge-item"><span style="color: #F59E0B;">●</span> Qdrant & FAISS Vector DB</div>
</div>
""").strip(), unsafe_allow_html=True)

# THỬ NGHIỆM NHANH VỚI 2 BÀI HÁT MẪU
with st.expander("✨ Thử nghiệm nhanh với 2 bài hát thực tế từ cơ sở dữ liệu", expanded=False):
    st.html("<p style='color: #94A3B8; font-size: 0.95rem; font-family: Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, sans-serif;'>Bấm vào các nút bên dưới để kiểm tra ngay độ chính xác của hệ thống mà không cần thu âm:</p>")
    col_demo1, col_demo2 = st.columns(2)
    
    with col_demo1:
        st.markdown(dedent("""
        <div class="sample-card">
            <div>
                <span style="background: rgba(16, 185, 129, 0.2); color: #34D399; font-weight: 700; font-size: 0.75rem; padding: 3px 10px; border-radius: 9999px;">BÀI MẪU 1</span>
                <h4 style="margin: 8px 0 2px 0; color: #FFFFFF;">Đừng Làm Trái Tim Anh Đau</h4>
                <p style="margin: 0 0 10px 0; color: #38BDF8; font-size: 0.9rem;">Sơn Tùng M-TP</p>
                <div style="background: rgba(0,0,0,0.3); padding: 8px 12px; border-radius: 8px; font-size: 0.85rem; color: #CBD5E1; font-style: italic; margin-bottom: 12px;">
                    "Hình như trong lòng anh đã không còn hình bóng ai ngoài em đâu..."
                </div>
            </div>
        </div>
        """).strip(), unsafe_allow_html=True)
        if os.path.exists("dataset/snippets/abPmZCZZrFA.wav"):
            st.audio("dataset/snippets/abPmZCZZrFA.wav", format="audio/wav")
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("🔍 Test Lời Bài 1", use_container_width=True, key="demo_btn_lyr1"):
                with st.spinner("Đang tìm kiếm..."):
                    res = search_from_text("Hình như trong lòng anh đã không còn hình bóng ai ngoài em đâu")
                    display_lyrics_results(res, threshold=LYRICS_SEARCH_THRESHOLD)
        with c_btn2:
            if st.button("⚡ Test Audio Bài 1", use_container_width=True, key="demo_btn_aud1"):
                with open("dataset/snippets/abPmZCZZrFA.wav", "rb") as f:
                    _handle_hybrid_input(f.read())
                    
    with col_demo2:
        st.markdown(dedent("""
        <div class="sample-card">
            <div>
                <span style="background: rgba(6, 182, 212, 0.2); color: #38BDF8; font-weight: 700; font-size: 0.75rem; padding: 3px 10px; border-radius: 9999px;">BÀI MẪU 2</span>
                <h4 style="margin: 8px 0 2px 0; color: #FFFFFF;">Mất Kết Nối</h4>
                <p style="margin: 0 0 10px 0; color: #38BDF8; font-size: 0.9rem;">Dương Domic</p>
                <div style="background: rgba(0,0,0,0.3); padding: 8px 12px; border-radius: 8px; font-size: 0.85rem; color: #CBD5E1; font-style: italic; margin-bottom: 12px;">
                    "Có tham lam lắm không khi anh chỉ muốn mình em trong đầu..."
                </div>
            </div>
        </div>
        """).strip(), unsafe_allow_html=True)
        if os.path.exists("dataset/snippets/lRsaDQtYqAo.wav"):
            st.audio("dataset/snippets/lRsaDQtYqAo.wav", format="audio/wav")
        c_btn3, c_btn4 = st.columns(2)
        with c_btn3:
            if st.button("🔍 Test Lời Bài 2", use_container_width=True, key="demo_btn_lyr2"):
                with st.spinner("Đang tìm kiếm..."):
                    res = search_from_text("Có tham lam lắm không khi anh chỉ muốn mình em trong đầu")
                    display_lyrics_results(res, threshold=LYRICS_SEARCH_THRESHOLD)
        with c_btn4:
            if st.button("⚡ Test Audio Bài 2", use_container_width=True, key="demo_btn_aud2"):
                with open("dataset/snippets/lRsaDQtYqAo.wav", "rb") as f:
                    _handle_hybrid_input(f.read())

st.html("<div style='margin-bottom: 25px;'></div>")


# ==========================================
# CHẾ ĐỘ 0: NHẬN DIỆN THÔNG MINH (HYBRID FUSION)
# ==========================================
if menu == "🌟 Nhận diện Thông minh (Hybrid)":
    st.html(dedent("""
    <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 16px; padding: 18px 24px; margin-bottom: 24px; font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.5rem;">🌟</span>
            <div>
                <b style="color: #34D399; font-size: 1.05rem;">Chế độ Dual-Modality Hybrid Fusion tối ưu nhất</b>
                <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 2px;">
                    Hát hoặc phát một đoạn nhạc (tối thiểu 5.0s). Hệ thống chạy đồng thời <b>Whisper (bóc tách câu hát)</b> và <b>CLAP (so khớp âm sắc)</b> để trả về kết quả chính xác nhất.
                </div>
            </div>
        </div>
    </div>
    """).strip())

    tab_mic_h, tab_upload_h = st.tabs(["🎙️ Thu âm qua Micro", "📁 Tải file Audio lên"])

    with tab_mic_h:
        st.html("<div style='text-align: center; margin: 20px 0 10px 0; color: #94A3B8; font-family: Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, sans-serif;'>Chạm vào biểu tượng Micro bên dưới để bắt đầu thu âm:</div>")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            hybrid_bytes = audio_recorder(text="", recording_color="#EF4444", neutral_color="#10B981", icon_name="microphone-lines", icon_size="5x", key="mic_hybrid")
        if hybrid_bytes:
            st.html("<br>")
            st.audio(hybrid_bytes, format="audio/wav")
            if st.button("🚀 Kích Hoạt Nhận Diện Đa Phương Thức", type="primary", use_container_width=True, key="btn_mic_hybrid"):
                _handle_hybrid_input(hybrid_bytes)

    with tab_upload_h:
        st.html("<div style='text-align: center; margin: 15px 0 5px 0; color: #94A3B8; font-family: Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, sans-serif;'>Chọn hoặc kéo thả file âm thanh đoạn hát (WAV, MP3):</div>")
        uploaded_hybrid = st.file_uploader("Định dạng hỗ trợ: WAV, MP3", type=["wav", "mp3"], key="uploader_hybrid", label_visibility="collapsed")
        if uploaded_hybrid is not None:
            st.audio(uploaded_hybrid, format="audio/wav")
            if st.button("🚀 Phân Tích Thông Minh File Audio", type="primary", use_container_width=True, key="btn_file_hybrid"):
                _handle_hybrid_input(uploaded_hybrid.getvalue())

# ==========================================
# CHẾ ĐỘ 1: CHỈ NHẬN DIỆN QUA LỜI HÁT (ASR)
# ==========================================
elif menu == "🎤 Chỉ tìm qua Lời hát (ASR)":
    st.html(dedent("""
    <div style="background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.25); border-radius: 16px; padding: 18px 24px; margin-bottom: 24px; font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.5rem;">🎤</span>
            <div>
                <b style="color: #38BDF8; font-size: 1.05rem;">Nhận diện qua Lời bài hát (Whisper ASR + SBERT FAISS)</b>
                <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 2px;">
                    Hát hoặc đọc một vài câu lời bài hát tiếng Việt. Whisper sẽ chuyển giọng nói thành văn bản tiếng Việt và tìm kiếm ngữ nghĩa theo từng phân đoạn.
                </div>
            </div>
        </div>
    </div>
    """).strip())

    tab_mic, tab_upload = st.tabs(["🎙️ Thu âm qua Micro", "📁 Tải file Audio lên"])

    with tab_mic:
        st.html("<div style='text-align: center; margin: 20px 0 10px 0; color: #94A3B8; font-family: Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, sans-serif;'>Chạm vào biểu tượng Micro để bắt đầu hát / đọc lời:</div>")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            audio_bytes = audio_recorder(text="", recording_color="#EF4444", neutral_color="#06B6D4", icon_name="microphone", icon_size="5x", key="mic_lyrics")
        if audio_bytes:
            st.html("<br>")
            st.audio(audio_bytes, format="audio/wav")
            if st.button("🚀 Nhận Dạng Lời Hát", type="primary", use_container_width=True, key="btn_mic_lyrics"):
                _handle_lyrics_input(audio_bytes)

    with tab_upload:
        st.html("<div style='text-align: center; margin: 15px 0 5px 0; color: #94A3B8; font-family: Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, sans-serif;'>Tải lên đoạn ghi âm câu hát (tối thiểu 5.0s):</div>")
        uploaded_file = st.file_uploader("Định dạng hỗ trợ: WAV, MP3", type=["wav", "mp3"], key="uploader_lyrics", label_visibility="collapsed")
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            if st.button("🚀 Nhận Dạng File Lời Hát", type="primary", use_container_width=True, key="btn_file_lyrics"):
                _handle_lyrics_input(uploaded_file.getvalue())

# ==========================================
# CHẾ ĐỘ 2: TÌM QUA LỜI NHẠC GÕ TAY
# ==========================================
elif menu == "✍️ Tìm qua Lời gõ tay":
    st.html(dedent("""
    <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 16px; padding: 18px 24px; margin-bottom: 24px; font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.5rem;">✍️</span>
            <div>
                <b style="color: #A5B4FC; font-size: 1.05rem;">Tìm kiếm Ngữ nghĩa Lời bài hát (Semantic Search)</b>
                <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 2px;">
                    Nhập bất kỳ câu hát tiếng Việt nào bạn nhớ (dù đúng một phần hay sai vài từ, AI vẫn tìm ra bài hát tương ứng).
                </div>
            </div>
        </div>
    </div>
    """).strip())

    user_query = st.text_input("Gõ câu hát bạn nhớ vào đây:", placeholder="Ví dụ: Hình như trong lòng anh đã không còn hình bóng ai ngoài em đâu...")
    if st.button("🔍 Khám Phá Bài Hát Ngay", type="primary", use_container_width=True):
        if len(user_query.strip()) < 3:
            st.warning("⚠️ Vui lòng nhập lời bài hát dài hơn một chút (ít nhất 3 ký tự)!")
        else:
            with st.spinner('⚡ Đang đối soát vector ngữ nghĩa trên 14,982 phân rã câu hát...'):
                results = search_from_text(user_query)
                display_lyrics_results(results, threshold=LYRICS_SEARCH_THRESHOLD)
# ==========================================
# CHẾ ĐỘ 3: THU THẬP NHẠC MỚI (AI DATA AGENT)
# ==========================================
elif menu == "📥 Thu thập Nhạc Mới (AI Agent)":
    st.html(dedent("""
    <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 16px; padding: 20px 24px; margin-bottom: 24px; font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.8rem;">🤖</span>
            <div>
                <b style="color: #34D399; font-size: 1.15rem;">AI Data Agent — Tự Động Quét & Nạp Thêm Nhạc Việt</b>
                <div style="color: #94A3B8; font-size: 0.92rem; margin-top: 3px;">
                    Agent tự động quét YouTube Music (BXH Trending, Top Videos VN hoặc Ca sĩ / Chủ đề bạn chỉ định), 
                    lọc tiếng Việt, tải 30s âm thanh, đồng bộ lời hát (LRCLIB), và tạo vector embeddings (Qdrant & FAISS).
                </div>
            </div>
        </div>
    </div>
    """).strip())

    col_cfg1, col_cfg2 = st.columns([1, 1])
    with col_cfg1:
        crawl_limit = st.slider("🎯 Số lượng bài hát mới muốn thu thập:", min_value=3, max_value=50, value=10, step=1)
        crawl_source = st.selectbox(
            "🌐 Nguồn dữ liệu:",
            ["Tất cả (BXH Trending + Tìm kiếm)", "Chỉ Bảng Xếp Hạng (Charts)", "Chỉ Tìm Kiếm (Search)"]
        )
    with col_cfg2:
        custom_artist = st.text_input(
            "🔍 Chủ đề / Tên ca sĩ muốn nạp (tùy chọn):",
            placeholder="Ví dụ: Anh Trai Say Hi, Sơn Tùng M-TP, Đen Vâu, Vũ, Rap Việt..."
        )
        st.caption("💡 Mẹo: Nhập tên ca sĩ bạn thích để Agent gom hàng loạt bài hát của nghệ sĩ đó vào kho nhạc.")

    source_map = {
        "Tất cả (BXH Trending + Tìm kiếm)": "all",
        "Chỉ Bảng Xếp Hạng (Charts)": "charts",
        "Chỉ Tìm Kiếm (Search)": "search"
    }

    if st.button("🚀 Khởi Chạy AI Data Agent", type="primary", use_container_width=True):
        with st.spinner("🤖 AI Data Agent đang hoạt động... Đang quét và nạp dữ liệu (có thể mất 1-3 phút)..."):
            from pipeline.data_agent import MusicDataAgent
            agent = MusicDataAgent(dry_run=False)
            agent.run_full_pipeline(
                limit=crawl_limit,
                source=source_map[crawl_source],
                query=custom_artist.strip() if custom_artist.strip() else None
            )
            st.success(f"🎉 Hoàn tất! AI Data Agent đã nạp thêm nhạc mới vào hệ thống. Tổng số bài hát hiện tại: {tracks_col.count_documents({})}.")
            st.info("💡 Hệ thống đã cập nhật cả MongoDB, Vector Qdrant (Âm thanh) và Vector FAISS (Lời hát). Bạn có thể sang tab Tìm kiếm để trải nghiệm ngay!")

    st.markdown("---")
    st.markdown("#### 💻 Hoặc chạy bằng dòng lệnh Terminal:")
    st.code("""
# 1. Thu thập 20 bài mới bất kỳ từ BXH & Xu hướng:
./venv/bin/python pipeline/data_agent.py --limit 20

# 2. Thu thập 15 bài theo ca sĩ / chủ đề cụ thể:
./venv/bin/python pipeline/data_agent.py --limit 15 --query "Anh Trai Say Hi"
./venv/bin/python pipeline/data_agent.py --limit 15 --query "Sơn Tùng M-TP"
./venv/bin/python pipeline/data_agent.py --limit 15 --query "Đen Vâu"

# 3. Quét nhanh chỉ từ Bảng Xếp Hạng (không kèm tìm kiếm):
./venv/bin/python pipeline/data_agent.py --limit 20 --source charts
""", language="bash")

# ==========================================
# CHẾ ĐỘ 4: PHÂN TÍCH TÍN HIỆU ÂM THANH
# ==========================================
elif menu == "📊 Phân tích Tín hiệu Âm thanh":
    st.html(dedent("""
    <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 16px; padding: 18px 24px; margin-bottom: 24px; font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.5rem;">📊</span>
            <div>
                <b style="color: #FBBF24; font-size: 1.05rem;">Phân tích Tín hiệu Âm thanh Kỹ thuật số (DSP)</b>
                <div style="color: #94A3B8; font-size: 0.9rem; margin-top: 2px;">
                    Trích xuất & trực quan hóa 7 đặc trưng miền tần số và thời gian (Spectrogram, MFCC, Chromagram, Spectral Contrast,...).
                </div>
            </div>
        </div>
    </div>
    """).strip())

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
# CHẾ ĐỘ 4: ĐÁNH GIÁ WER/CER
# ==========================================
elif menu == "📈 Đánh giá WER/CER":
    st.html('<div style="font-size: 2rem; font-weight: 800; color: #FFFFFF; font-family: Plus Jakarta Sans, -apple-system, sans-serif;">Đánh giá WER/CER</div>')
    st.html('<div style="color: #94A3B8; font-size: 1.1rem; margin-bottom: 16px; font-family: Plus Jakarta Sans, -apple-system, sans-serif;">Phân tích chất lượng nhận diện giọng nói của Whisper trên dataset Tiếng Việt</div>')

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
        st.markdown("### 📊 Thống kê tổng quan (Tiếng Việt)")
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
        ax_hist.set_title('Phân bố WER trên dataset Tiếng Việt')
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

        # Top best / worst
        import pandas as pd
        
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