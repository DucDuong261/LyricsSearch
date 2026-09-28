"""
ui_components.py — Reusable UI Rendering Components
=====================================================
Trích xuất các hàm hiển thị bị lặp lại trong demoweb.py thành module tái sử dụng.

Các hàm chính:
  - save_audio_to_temp()           — Lưu audio bytes → temp file
  - render_lyrics_highlight()      — Highlight câu khớp trong lời bài hát
  - display_hybrid_results()       — Render toàn bộ kết quả Hybrid Fusion (CLAP + SBERT)
  - display_lyrics_results()       — Render kết quả tìm kiếm qua lời bài hát
"""

import os
import tempfile
from textwrap import dedent
import streamlit as st


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
        try:
            os.remove(file_path)
        except OSError:
            pass


# Font family constant để tránh lặp lại
_FONT = "'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif"
_FONT_MONO = "'JetBrains Mono', monospace"


def _render_html(html_content):
    """
    Render raw HTML sử dụng st.html() (Streamlit >= 1.33).
    Streamlit 1.63+ không còn hỗ trợ HTML phức tạp qua st.markdown(unsafe_allow_html=True).
    """
    st.html(html_content)


# ==========================================
# 2. RENDER LYRICS HIGHLIGHT
# ==========================================
def render_lyrics_highlight(full_lyrics, matched_line, expanded=False):
    """
    Hiển thị toàn bộ lời bài hát dạng Music Lyric Sheet với câu khớp sáng rực.
    """
    if not full_lyrics:
        return

    with st.expander("📜 Xem toàn bộ lời bài hát (Xác thực ngữ cảnh)", expanded=expanded):
        lines = full_lyrics.split('\n')
        clean_matched = matched_line.strip().lower() if matched_line else ""
        highlighted_lines = []

        for idx, line in enumerate(lines, 1):
            clean_line = line.strip()
            clean_lower = clean_line.lower()

            is_match = False
            if clean_lower and len(clean_lower) >= 4:
                if clean_lower in clean_matched or clean_matched in clean_lower:
                    is_match = True
            elif clean_lower and clean_lower == clean_matched:
                is_match = True

            line_num_str = f'<span style="color: #475569; font-size: 0.8rem; user-select: none; width: 32px; display: inline-block; font-family: {_FONT};">{idx:02d}</span>'

            if is_match:
                highlighted_lines.append(
                    f'<div style="background: linear-gradient(90deg, rgba(16, 185, 129, 0.22) 0%, rgba(6, 182, 212, 0.12) 100%); '
                    f'border-left: 3px solid #10B981; padding: 4px 10px; margin: 3px 0; border-radius: 0 8px 8px 0; font-weight: 700; color: #34D399;">'
                    f'{line_num_str}🎯 <b>{line}</b></div>'
                )
            else:
                highlighted_lines.append(
                    f'<div style="padding: 2px 10px; color: #94A3B8; font-size: 0.95rem;">'
                    f'{line_num_str}{line}</div>'
                )

        highlighted_html = "".join(highlighted_lines)
        html_box = f'<div style="background: rgba(10, 14, 23, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px; max-height: 420px; overflow-y: auto; font-family: {_FONT_MONO};">{highlighted_html}</div>'
        _render_html(html_box)


# ==========================================
# 3. RENDER KẾT QUẢ HYBRID (CLAP + SBERT)
# ==========================================
def display_hybrid_results(results, transcription, msg):
    """
    Render toàn bộ kết quả tìm kiếm Hybrid Fusion phong cách Spotify Pro.
    """
    if msg.startswith("⚠️"):
        st.warning(msg)
        return

    if not results:
        st.error("❌ Không tìm thấy bài hát phù hợp trong cơ sở dữ liệu!")
        return

    top1 = results[0]
    final_score = top1.get('final_score', 0)
    lyrics_score = top1.get('lyrics_score', 0)
    audio_score = top1.get('audio_score', 0)

    score_pct = int(min(100, max(0, final_score * 100)))
    if score_pct >= 85:
        badge_text = "ĐỘ CHÍNH XÁC XUẤT SẮC"
    elif score_pct >= 65:
        badge_text = "ĐỘ CHÍNH XÁC CAO"
    else:
        badge_text = "KẾT QUẢ KHẢ DĨ"

    card_html = dedent(f"""
    <div style="
        background: linear-gradient(145deg, rgba(23, 32, 54, 0.9) 0%, rgba(13, 18, 30, 0.95) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 20px;
        padding: 26px;
        box-shadow: 0 20px 45px -10px rgba(0, 0, 0, 0.6), 0 0 25px rgba(16, 185, 129, 0.12);
        margin: 20px 0;
        font-family: {_FONT};
    ">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <span style="
                    background: rgba(16, 185, 129, 0.18);
                    color: #34D399;
                    border: 1px solid rgba(16, 185, 129, 0.35);
                    padding: 5px 14px;
                    border-radius: 9999px;
                    font-size: 0.78rem;
                    font-weight: 700;
                    letter-spacing: 0.05em;
                ">🌟 TOP 1 • {badge_text}</span>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 2rem; font-weight: 800; color: #10B981;">{score_pct}%</span>
                <span style="font-size: 0.8rem; color: #94A3B8; display: block;">Độ tin cậy tổng hợp</span>
            </div>
        </div>

        <div style="margin-top: 14px;">
            <h2 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">
                {top1['title']}
            </h2>
            <p style="margin: 6px 0 16px 0; font-size: 1.15rem; color: #38BDF8; font-weight: 500;">
                👤 Ca sĩ / Nghệ sĩ: <span style="color: #F8FAFC; font-weight: 600;">{top1['artist']}</span>
            </p>
        </div>

        <div style="background: rgba(0,0,0,0.35); padding: 14px; border-radius: 12px; margin: 16px 0; border: 1px solid rgba(255,255,255,0.05);">
            <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 8px;">
                🔗 <b>Cơ chế dung hợp:</b> <span style="color: #E2E8F0;">{top1.get('fusion_desc', 'Dung hợp thích ứng')}</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 10px;">
                <div style="background: rgba(255,255,255,0.03); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="display: flex; justify-content: space-between; font-size: 0.82rem;">
                        <span style="color: #A7F3D0;">🎙️ Nhánh Lời (SBERT)</span>
                        <span style="font-weight: 700; color: #34D399;">{lyrics_score*100:.1f}%</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); border-radius: 4px; height: 6px; margin-top: 6px; overflow: hidden;">
                        <div style="background: #10B981; width: {min(100, int(lyrics_score*100))}%; height: 100%;"></div>
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="display: flex; justify-content: space-between; font-size: 0.82rem;">
                        <span style="color: #BAE6FD;">🎧 Nhánh Audio (CLAP)</span>
                        <span style="font-weight: 700; color: #38BDF8;">{audio_score*100:.1f}%</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); border-radius: 4px; height: 6px; margin-top: 6px; overflow: hidden;">
                        <div style="background: #06B6D4; width: {min(100, int(audio_score*100))}%; height: 100%;"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """).strip()
    _render_html(card_html)

    # Audio player
    audio_file = top1.get('audio_path', '')
    if audio_file and os.path.exists(audio_file):
        st.markdown("##### 🎧 Bản thu 30 giây gốc trong cơ sở dữ liệu:")
        st.audio(audio_file, format="audio/wav")

    # Transcription
    trans = transcription
    if trans and trans != "Nhiễu/Không nghe rõ" and not trans.startswith("Lỗi"):
        trans_html = dedent(f"""
        <div style="background: rgba(14, 165, 233, 0.1); border-left: 3px solid #0EA5E9; padding: 12px 16px; border-radius: 8px; margin: 14px 0; font-family: {_FONT};">
            <span style="color: #7DD3FC; font-size: 0.85rem; font-weight: 600;">🎙️ AI NHẬN DẠNG TIẾNG NÓI (WHISPER ASR):</span>
            <div style="color: #F0F9FF; font-size: 1.1rem; font-style: italic; margin-top: 4px;">"{trans}"</div>
        </div>
        """).strip()
        _render_html(trans_html)

        if top1.get('full_lyrics'):
            render_lyrics_highlight(
                top1['full_lyrics'],
                top1.get('matched_line', ''),
                expanded=False
            )
    else:
        st.caption("ℹ️ Không phát hiện câu từ rõ nghĩa — Hệ thống đã tự động dồn trọng số nhận diện sang nhánh Audio Encoder (CLAP).")

    # Alternate candidates
    if len(results) > 1:
        _render_html("<h4 style='margin-top: 24px; color: #CBD5E1; font-family: " + _FONT + ";'>📑 Các bài hát có đặc trưng tương đồng khác:</h4>")
        for idx, res in enumerate(results[1:], start=2):
            res_score_pct = int(min(100, max(0, res.get('final_score', 0) * 100)))
            r_audio = res.get('audio_path', '')

            cand_html = dedent(f"""
            <div style="
                background: rgba(255, 255, 255, 0.025);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 12px;
                padding: 12px 18px;
                margin: 8px 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-family: {_FONT};
            ">
                <div>
                    <span style="color: #64748B; font-weight: 700; margin-right: 8px;">#{idx}</span>
                    <b style="color: #F1F5F9; font-size: 1.05rem;">{res['title']}</b>
                    <span style="color: #94A3B8; font-size: 0.95rem; margin-left: 6px;">— {res['artist']}</span>
                    <div style="color: #64748B; font-size: 0.8rem; margin-top: 2px;">{res.get('fusion_desc', '')}</div>
                </div>
                <div style="text-align: right;">
                    <span style="background: rgba(255,255,255,0.08); padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 700; color: #CBD5E1;">
                        {res_score_pct}%
                    </span>
                </div>
            </div>
            """).strip()
            _render_html(cand_html)

            if r_audio and os.path.exists(r_audio):
                with st.expander(f"🎧 Nghe thử bài #{idx}: {res['title']}"):
                    st.audio(r_audio, format="audio/wav")


# ==========================================
# 4. RENDER KẾT QUẢ LYRICS SEARCH
# ==========================================
def display_lyrics_results(results, predicted_text=None, threshold=0.60):
    """
    Render kết quả tìm kiếm qua lời bài hát (Text / ASR).
    """
    if predicted_text:
        if predicted_text.startswith("⚠️"):
            st.warning(predicted_text)
            return
        elif predicted_text == "Nhiễu/Không nghe rõ":
            st.error("⚠️ AI không nghe rõ lời bài hát hoặc đoạn thu âm quá ồn. Vui lòng thử lại gần micro hơn!")
            return
        else:
            asr_html = dedent(f"""
            <div style="background: rgba(16, 185, 129, 0.1); border-left: 3px solid #10B981; padding: 12px 16px; border-radius: 8px; margin: 12px 0; font-family: {_FONT};">
                <span style="color: #6EE7B7; font-size: 0.85rem; font-weight: 600;">🎙️ AI NHẬN DẠNG TIẾNG NÓI (WHISPER):</span>
                <div style="color: #F0FDF4; font-size: 1.1rem; font-style: italic; margin-top: 4px;">"{predicted_text}"</div>
            </div>
            """).strip()
            _render_html(asr_html)

    if not results:
        st.error("❌ Rất tiếc, hệ thống không tìm thấy bài hát nào khớp với câu hát này!")
        return

    best_score = results[0]['score']

    if best_score < threshold:
        st.error(
            f"❌ Không tìm thấy bài hát phù hợp! Dữ liệu đầu vào quá lạ "
            f"(Độ tin cậy cao nhất chỉ đạt {best_score:.2f}, "
            f"thấp hơn ngưỡng yêu cầu của hệ thống là {threshold})."
        )
        return

    top1 = results[0]
    score_pct = int(min(100, max(0, best_score * 100)))

    lyr_card_html = dedent(f"""
    <div style="
        background: linear-gradient(145deg, rgba(23, 32, 54, 0.9) 0%, rgba(13, 18, 30, 0.95) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 20px 45px -10px rgba(0, 0, 0, 0.6), 0 0 25px rgba(16, 185, 129, 0.12);
        margin: 20px 0;
        font-family: {_FONT};
    ">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <span style="
                background: rgba(16, 185, 129, 0.2);
                color: #34D399;
                border: 1px solid rgba(16, 185, 129, 0.4);
                padding: 4px 12px;
                border-radius: 9999px;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.05em;
            ">🎯 KẾT QUẢ KHỚP CAO NHẤT</span>
            <div style="text-align: right;">
                <span style="font-size: 2rem; font-weight: 800; color: #10B981;">{score_pct}%</span>
                <span style="font-size: 0.8rem; color: #94A3B8; display: block;">Độ tương đồng</span>
            </div>
        </div>

        <h2 style="margin: 12px 0 4px 0; font-size: 2.2rem; font-weight: 800; color: #FFFFFF;">
            {top1['title']}
        </h2>
        <p style="margin: 0 0 16px 0; font-size: 1.15rem; color: #38BDF8; font-weight: 500;">
            👤 Ca sĩ: <span style="color: #F8FAFC; font-weight: 600;">{top1['artist']}</span>
        </p>

        <div style="
            background: rgba(10, 14, 23, 0.75);
            border-left: 3px solid #10B981;
            padding: 12px 16px;
            border-radius: 8px;
            margin: 14px 0;
            border: 1px solid rgba(255,255,255,0.04);
        ">
            <span style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Câu hát khớp trong bài:</span>
            <div style="color: #34D399; font-size: 1.15rem; font-weight: 600; margin-top: 4px; font-style: italic;">
                "{top1['matched_line']}"
            </div>
        </div>
    </div>
    """).strip()
    _render_html(lyr_card_html)

    # Audio player
    audio_file = top1.get('audio_path', '')
    if audio_file and os.path.exists(audio_file):
        st.markdown("##### 🎧 Bản thu 30 giây gốc trong cơ sở dữ liệu:")
        st.audio(audio_file, format="audio/wav")

    # Lyrics Highlight
    render_lyrics_highlight(
        top1.get('full_lyrics', ''),
        top1.get('matched_line', ''),
        expanded=False
    )

    # Runner ups
    if len(results) > 1:
        _render_html("<h4 style='margin-top: 24px; color: #CBD5E1; font-family: " + _FONT + ";'>📑 Các bài hát có câu từ tương tự:</h4>")
        for idx, res in enumerate(results[1:], start=2):
            if res['score'] >= threshold - 0.15:
                r_pct = int(min(100, max(0, res['score'] * 100)))
                cand_lyr_html = dedent(f"""
                <div style="
                    background: rgba(255, 255, 255, 0.025);
                    border: 1px solid rgba(255, 255, 255, 0.06);
                    border-radius: 12px;
                    padding: 12px 18px;
                    margin: 8px 0;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-family: {_FONT};
                ">
                    <div>
                        <span style="color: #64748B; font-weight: 700; margin-right: 8px;">#{idx}</span>
                        <b style="color: #F1F5F9; font-size: 1.05rem;">{res['title']}</b> — <span style="color: #94A3B8;">{res['artist']}</span>
                        <div style="color: #64748B; font-size: 0.82rem; font-style: italic; margin-top: 2px;">"{res.get('matched_line', '')[:85]}..."</div>
                    </div>
                    <div>
                        <span style="background: rgba(255,255,255,0.08); padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 700; color: #CBD5E1;">
                            {r_pct}%
                        </span>
                    </div>
                </div>
                """).strip()
                _render_html(cand_lyr_html)
