"""
ui_components.py — Reusable UI Rendering Components
=====================================================
Trích xuất các hàm hiển thị bị lặp lại trong demoweb.py thành module tái sử dụng.

Các hàm chính:
  - save_audio_to_temp()           — Lưu audio bytes → temp file
  - render_lyrics_highlight()      — Highlight câu khớp trong lời bài hát
  - display_hybrid_results()       — Render toàn bộ kết quả Hybrid Fusion
  - display_qbh_results()          — Render toàn bộ kết quả Query by Humming
  - display_lyrics_results()       — Render kết quả tìm kiếm qua lời bài hát
"""

import os
import tempfile
import streamlit as st
import matplotlib.pyplot as plt
from melody_engine import plot_melody_alignment


# ==========================================
# 1. TIỆN ÍCH CHUNG (UTILITIES)
# ==========================================
def save_audio_to_temp(audio_bytes, suffix=".wav"):
    """
    Lưu audio bytes (từ recorder hoặc uploader) vào file tạm.
    Trả về đường dẫn file tạm (cần xóa sau khi dùng xong).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(audio_bytes)
        return tmp_file.name


def cleanup_temp_file(file_path):
    """Xóa file tạm nếu tồn tại."""
    if file_path and os.path.exists(file_path):
        os.remove(file_path)


# ==========================================
# 2. RENDER LYRICS HIGHLIGHT (DRY: trước đó lặp 3 lần)
# ==========================================
def render_lyrics_highlight(full_lyrics, matched_line, expanded=False):
    """
    Hiển thị toàn bộ lời bài hát với câu khớp được highlight xanh lá.

    Args:
        full_lyrics: Lời bài hát đầy đủ (string, phân cách bằng \\n)
        matched_line: Câu hát khớp nhất (string)
        expanded: Mở sẵn expander hay không
    """
    if not full_lyrics:
        return

    with st.expander("📄 Xác thực bằng cách xem toàn bộ lời bài hát", expanded=expanded):
        lines = full_lyrics.split('\n')
        clean_matched = matched_line.strip().lower() if matched_line else ""
        highlighted_lines = []

        for line in lines:
            clean_line = line.strip()
            clean_lower = clean_line.lower()

            # Khớp chính xác hoặc khớp chuỗi con có độ dài hợp lý
            is_match = False
            if clean_lower and len(clean_lower) >= 4:
                if clean_lower in clean_matched or clean_matched in clean_lower:
                    is_match = True
            elif clean_lower and clean_lower == clean_matched:
                is_match = True

            if is_match:
                highlighted_lines.append(
                    f'<mark style="background-color: rgba(29, 185, 84, 0.35); '
                    f'color: var(--text-color); font-weight: bold; '
                    f'padding: 2px 6px; border-radius: 4px;">{line}</mark>'
                )
            else:
                highlighted_lines.append(line)

        highlighted_html = "<br>".join(highlighted_lines)
        st.markdown(
            f'<div style="line-height: 1.8; color: var(--text-color); '
            f'font-size: 1.05rem;">{highlighted_html}</div>',
            unsafe_allow_html=True
        )


# ==========================================
# 3. RENDER KẾT QUẢ HYBRID (DRY: trước đó lặp 2 lần)
# ==========================================
def display_hybrid_results(results, transcription, query_feats, msg):
    """
    Render toàn bộ kết quả tìm kiếm Hybrid Fusion.
    Bao gồm: Result Card + Audio Player + Lyrics Highlight + Pitch Chart + References.

    Args:
        results: Danh sách kết quả từ search_hybrid()
        transcription: Văn bản Whisper nhận diện được
        query_feats: Đặc trưng giai điệu query (từ melody engine)
        msg: Thông báo trạng thái từ search_hybrid()
    """
    if msg.startswith("⚠️"):
        st.warning(msg)
        return

    if not results:
        st.error("Không tìm thấy bài hát phù hợp!")
        return

    top1 = results[0]

    # ---- Result Card ----
    st.markdown(f"""
    <div class="result-card">
        <span class="badge-hybrid">🌟 KẾT QUẢ DUNG HỢP TỐI ƯU</span>
        <h2 class="song-title">{top1['title']}</h2>
        <p class="song-artist">{top1['artist']}</p>
        <div class="lyric-box">🔗 <b>Cơ chế dung hợp:</b> {top1['fusion_desc']}</div>
        <p class="score-text">Độ tin cậy tổng hợp: <b>{top1['final_score']*100:.1f}%</b> | Điểm Lời: {top1['lyrics_score']*100:.1f}% | Điểm Giai điệu: {top1['melody_score']*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Audio Player ----
    audio_file = top1.get('audio_path', '')
    if os.path.exists(audio_file):
        st.markdown("##### 🎧 Bản thu gốc:")
        st.audio(audio_file, format="audio/wav")

    # ---- Bằng chứng 1: Lời bài hát ----
    trans = transcription
    if trans and trans != "Nhiễu/Không nghe rõ" and not trans.startswith("Lỗi"):
        st.markdown(f"🎙️ **AI nghe được câu hát:** *\"{trans}\"*")
        if top1.get('full_lyrics'):
            render_lyrics_highlight(
                top1['full_lyrics'],
                top1.get('matched_line', ''),
                expanded=False
            )
    else:
        st.caption("ℹ️ Không phát hiện câu từ rõ nghĩa — Hệ thống đã tự động dồn trọng số nhận diện sang nhánh Giai điệu (QBH).")

    # ---- Bằng chứng 2: Giai điệu cao độ ----
    if top1.get('ref_entry') is not None and query_feats is not None:
        with st.expander("📈 Xem Biểu đồ Dóng hàng Cao độ Giai điệu (Pitch Alignment)", expanded=True):
            fig = plot_melody_alignment(query_feats, top1)
            st.pyplot(fig)
            plt.close(fig)

    # ---- Kết quả tham khảo ----
    st.divider()
    st.markdown("#### 🎧 Các kết quả tham khảo khác")
    for idx, res in enumerate(results[1:], start=2):
        st.write(
            f"**Top {idx}: {res['title']}** - {res['artist']} "
            f"*(Độ tin cậy: {res['final_score']*100:.1f}% | {res['fusion_desc']})*"
        )


# ==========================================
# 4. RENDER KẾT QUẢ QBH (DRY: trước đó lặp 2 lần)
# ==========================================
def display_qbh_results(results, query_feats, msg):
    """
    Render toàn bộ kết quả tìm kiếm Query by Humming.
    Bao gồm: Result Card + Audio Player + Pitch Chart + References.

    Args:
        results: Danh sách kết quả từ search_by_melody()
        query_feats: Đặc trưng giai điệu query
        msg: Thông báo trạng thái
    """
    if msg.startswith("⚠️"):
        st.warning(msg)
        return

    if not results:
        st.error("Không tìm thấy giai điệu khớp trong cơ sở dữ liệu!")
        return

    top1 = results[0]

    # ---- Result Card ----
    st.markdown(f"""
    <div class="result-card">
        <span class="badge-qbh">GIAI ĐIỆU KHỚP NHẤT</span>
        <h2 class="song-title">{top1['title']}</h2>
        <p class="song-artist">{top1['artist']}</p>
        <div class="lyric-box">🎧 Khớp với đoạn {top1['matched_start_sec']:.1f}s - {top1['matched_end_sec']:.1f}s trong bản thu gốc</div>
        <p class="score-text">Độ tương đồng giai điệu: <b>{top1['score']*100:.1f}%</b> (DTW Normalized Cost: {top1['dtw_cost']:.4f})</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Audio Player ----
    audio_file = top1.get('audio_path', '')
    if os.path.exists(audio_file):
        st.markdown("##### 🎧 Bản thu gốc:")
        st.audio(audio_file, format="audio/wav")

    # ---- Biểu đồ Pitch Alignment ----
    if query_feats is not None:
        with st.expander("📈 Xem Biểu đồ Dóng hàng Cao độ Giai điệu (Pitch Alignment)", expanded=True):
            fig = plot_melody_alignment(query_feats, top1)
            st.pyplot(fig)
            plt.close(fig)

    # ---- Kết quả tham khảo ----
    st.divider()
    st.markdown("#### 🎧 Các kết quả giai điệu tương tự")
    for idx, res in enumerate(results[1:], start=2):
        st.write(
            f"**Top {idx}: {res['title']}** - {res['artist']} "
            f"*(Độ tương đồng: {res['score']*100:.1f}% | "
            f"Đoạn: {res['matched_start_sec']:.1f}s - {res['matched_end_sec']:.1f}s)*"
        )


# ==========================================
# 5. RENDER KẾT QUẢ LYRICS SEARCH
# ==========================================
def display_lyrics_results(results, predicted_text=None, threshold=0.60):
    """
    Render kết quả tìm kiếm qua lời bài hát (Text / ASR).

    Args:
        results: Danh sách kết quả từ search_from_text()
        predicted_text: Văn bản Whisper nhận diện (None nếu tìm bằng text gõ tay)
        threshold: Ngưỡng tối thiểu để hiển thị kết quả
    """
    if predicted_text:
        if predicted_text.startswith("⚠️"):
            st.warning(predicted_text)
            return
        elif predicted_text == "Nhiễu/Không nghe rõ":
            st.error("⚠️ AI không nghe rõ lời bài hát hoặc đoạn thu âm quá ồn. Vui lòng thử lại!")
            return
        else:
            st.info(f"🎙️ **AI nghe được:** *\"{predicted_text}\"*")

    if not results:
        st.error("Rất tiếc, hệ thống không trả về kết quả nào!")
        return

    best_score = results[0]['score']

    # CHỈ CHO PHÉP HIỂN THỊ NẾU ĐIỂM SỐ VƯỢT MỨC CHUẨN
    if best_score < threshold:
        st.error(
            f"❌ Không tìm thấy bài hát phù hợp! Dữ liệu đầu vào quá lạ "
            f"(Độ tin cậy cao nhất chỉ đạt {best_score:.2f}, "
            f"thấp hơn tiêu chuẩn hệ thống là {threshold})."
        )
        return

    top1 = results[0]

    # ---- Result Card ----
    st.markdown(f"""
    <div class="result-card">
        <span class="badge-top1">KẾT QUẢ PHÙ HỢP NHẤT</span>
        <h2 class="song-title">{top1['title']}</h2>
        <p class="song-artist">{top1['artist']}</p>
        <div class="lyric-box">"{top1['matched_line']}"</div>
        <p class="score-text">Độ tin cậy: {best_score:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Audio Player ----
    audio_file = top1.get('audio_path', '')
    if os.path.exists(audio_file):
        st.audio(audio_file, format="audio/wav")
    else:
        st.warning("⚠️ Không tìm thấy file gốc trên máy để phát.")

    # ---- Lyrics Highlight ----
    render_lyrics_highlight(
        top1.get('full_lyrics', ''),
        top1.get('matched_line', ''),
        expanded=False
    )

    # ---- Kết quả tham khảo ----
    st.divider()
    st.markdown("#### 🎧 Các kết quả tham khảo")
    for idx, res in enumerate(results[1:], start=2):
        if res['score'] >= threshold - 0.1:
            st.write(
                f"**Top {idx}: {res['title']}** - {res['artist']} "
                f"*(Score: {res['score']:.4f})*"
            )
