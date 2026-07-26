import streamlit as st
import faiss
import json
import numpy as np
import torch
import librosa
import os
import tempfile
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from audio_recorder_streamlit import audio_recorder

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
    .song-title { font-size: 1.8rem; font-weight: 800; margin: 12px 0 4px 0; color: var(--text-color); }
    .song-artist { font-size: 1.1rem; color: var(--text-color); opacity: 0.8; margin: 0 0 16px 0; }
    .lyric-box { background-color: var(--background-color); border-left: 4px solid #1DB954; padding: 12px; border-radius: 4px; font-style: italic; color: var(--text-color); }
    .score-text { color: var(--text-color); opacity: 0.6; font-size: 0.85rem; margin-top: 15px; margin-bottom: 0px; }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. LOAD AI
# ==========================================
@st.cache_resource(show_spinner="Đang đánh thức bộ não AI...")
def load_all_systems():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    whisper_model = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=device,
        generate_kwargs={"task": "transcribe"}
    )
    sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    index = faiss.read_index("lyrics_db.index")
    with open('id_mapping.json', 'r', encoding='utf-8') as f:
        chunk_mapping = json.load(f)
    client = MongoClient("mongodb://localhost:27017/")
    tracks_col = client["lyrics_recognition"]["tracks"]
    return whisper_model, sbert_model, index, chunk_mapping, tracks_col

whisper_model, sbert_model, index, chunk_mapping, tracks_col = load_all_systems()

# ==========================================
# 2. LOGIC XỬ LÝ LÕI
# ==========================================
def search_from_text(query_text, top_k=3):
    query_vector = sbert_model.encode([query_text], normalize_embeddings=True)
    distances, indices = index.search(np.array(query_vector).astype('float32'), 15)
    seen_ids = set()
    results = []
    for i in range(15):
        if len(results) >= top_k: break
        match_info = chunk_mapping[indices[0][i]]
        mongo_id = match_info["mongo_id"]
        if mongo_id in seen_ids: continue
        seen_ids.add(mongo_id)
        
        track = tracks_col.find_one({"_id": mongo_id})
        if track:
            # Lấy raw_path, nếu là None thì ép thành chuỗi rỗng ("")
            raw_path = track.get('snippet', {}).get('path')
            safe_path = raw_path.replace('\\', '/') if isinstance(raw_path, str) else ""

            results.append({
                "title": track['metadata']['title'],
                "artist": ", ".join(track['metadata']['artist']),
                "matched_line": match_info["matched_line"],
                "score": distances[0][i],
                "audio_path": safe_path,
                "full_lyrics": track.get('lyrics', {}).get('full_text', '') 
            })
    return results

def process_audio(audio_path):
    audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
    result = whisper_model({"array": audio_array, "sampling_rate": sampling_rate})
    text = result["text"].strip()
    
    # LỌC NHIỄU: Nếu Whisper nghe ra âm thanh vô nghĩa quá ngắn
    if not text or len(text) < 5: 
        return [], "Nhiễu/Không nghe rõ"
        
    return search_from_text(text), text

# THÊM BIẾN THRESHOLD = 0.60 LÀM CHUẨN ĐẦU VÀO
def display_results(results, predicted_text=None, threshold=0.60):
    if predicted_text:
        if predicted_text == "Nhiễu/Không nghe rõ":
            st.error("⚠️ AI không nghe rõ lời bài hát hoặc đoạn thu âm quá ồn. Vui lòng thử lại!")
            return
        else:
            st.info(f"🎙️ **AI nghe được:** *\"{predicted_text}\"*")
        
    if results:
        best_score = results[0]['score']
        
        # CHỈ CHO PHÉP HIỂN THỊ NẾU ĐIỂM SỐ VƯỢT MỨC CHUẨN 0.60
        if best_score >= threshold:
            top1 = results[0]
            st.markdown(f"""
            <div class="result-card">
                <span class="badge-top1">KẾT QUẢ PHÙ HỢP NHẤT</span>
                <h2 class="song-title">{top1['title']}</h2>
                <p class="song-artist">{top1['artist']}</p>
                <div class="lyric-box">"{top1['matched_line']}"</div>
                <p class="score-text">Độ tin cậy: {best_score:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            audio_file = top1.get('audio_path', '')
            if os.path.exists(audio_file):
                st.audio(audio_file, format="audio/wav")
            else:
                st.warning("⚠️ Không tìm thấy file gốc trên máy để phát.")
                
            full_lyrics = top1.get('full_lyrics', '')
            matched_line = top1.get('matched_line', '')
            
            if full_lyrics:
                with st.expander("📄 Xác thực bằng cách xem toàn bộ lời bài hát", expanded=False):
                    lines = full_lyrics.split('\n')
                    highlighted_lines = []
                    for line in lines:
                        clean_line = line.strip()
                        if clean_line and clean_line in matched_line:
                            highlighted_lines.append(f'<mark style="background-color: rgba(29, 185, 84, 0.35); color: var(--text-color); font-weight: bold; padding: 2px 6px; border-radius: 4px;">{line}</mark>')
                        else:
                            highlighted_lines.append(line)
                    
                    highlighted_html = "<br>".join(highlighted_lines)
                    st.markdown(f'<div style="line-height: 1.8; color: var(--text-color); font-size: 1.05rem;">{highlighted_html}</div>', unsafe_allow_html=True)
                
            st.divider()
            st.markdown("#### 🎧 Các kết quả tham khảo")
            for idx, res in enumerate(results[1:], start=2):
                if res['score'] >= threshold - 0.1: # Các kết quả phụ cũng phải đạt chuẩn
                    st.write(f"**Top {idx}: {res['title']}** - {res['artist']} *(Score: {res['score']:.4f})*")
        else:
            # BÁO LỖI VÀ HIỂN THỊ ĐIỂM THỰC TẾ NẾU DƯỚI NGƯỠNG
            st.error(f"❌ Không tìm thấy bài hát phù hợp! Dữ liệu đầu vào quá lạ (Độ tin cậy cao nhất chỉ đạt {best_score:.2f}, thấp hơn tiêu chuẩn hệ thống là {threshold}).")
    else:
        st.error("Rất tiếc, hệ thống không trả về kết quả nào!")

# ==========================================
# 3. GIAO DIỆN CHÍNH
# ==========================================
with st.sidebar:
    st.markdown("## 🎶 Điều hướng")
    st.markdown("---")
    menu = st.radio("", ["🎤 Nhận diện qua Micro", "📁 Tải file Audio", "✍️ Tìm qua Lời nhạc"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Đồ án: Nhận diện lời bài hát từ tín hiệu âm thanh.")

st.markdown('<div class="app-title">AI Music Matcher</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Tìm kiếm bài hát nhanh chóng qua âm thanh và lời nhạc</div>', unsafe_allow_html=True)

if menu == "🎤 Nhận diện qua Micro":
    st.markdown("<h4 style='text-align: center; margin-bottom: 20px;'>Chạm vào Micro để bắt đầu</h4>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        audio_bytes = audio_recorder(text="", recording_color="#FF4136", neutral_color="#1DB954", icon_name="microphone", icon_size="5x")
    if audio_bytes:
        st.markdown("<br>", unsafe_allow_html=True)
        st.audio(audio_bytes, format="audio/wav")
        if st.button("🚀 Xử Lý Bản Thu", type="primary", use_container_width=True):
            with st.spinner("AI đang bóc tách âm thanh..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name
                results, text = process_audio(tmp_path)
                display_results(results, text)
                os.remove(tmp_path)

elif menu == "📁 Tải file Audio":
    st.markdown("<h4 style='text-align: center;'>Tải lên đoạn điệp khúc</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Định dạng hỗ trợ: WAV, MP3", type=["wav", "mp3"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        if st.button("🚀 Phân Tích Audio", type="primary", use_container_width=True):
            with st.spinner("Đang xử lý tín hiệu..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                results, text = process_audio(tmp_path)
                display_results(results, text)
                os.remove(tmp_path)

elif menu == "✍️ Tìm qua Lời nhạc":
    st.markdown("<h4 style='text-align: center;'>Bạn nhớ được giai điệu nào?</h4>", unsafe_allow_html=True)
    user_query = st.text_input("Gõ trực tiếp vào đây:", placeholder="Ví dụ: Bên em bao lâu...")
    if st.button("🚀 Tìm Bài Hát", type="primary", use_container_width=True):
        if len(user_query.strip()) < 3:
            st.warning("Vui lòng nhập lời bài hát dài hơn một chút!")
        else:
            with st.spinner('Đang rà soát dữ liệu...'):
                results = search_from_text(user_query)
                display_results(results)