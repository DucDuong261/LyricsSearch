"""
Module B: Đánh giá WER/CER có hệ thống
========================================
So sánh văn bản Whisper nhận diện (predicted_text) với lời bài hát gốc (full_text)
cho toàn bộ dataset, phân tích lỗi và xuất báo cáo.

Sử dụng thư viện `jiwer` (chuẩn công nghiệp cho đánh giá ASR).
"""

import os
import re
import json
import unicodedata
from collections import Counter

from pymongo import MongoClient
from jiwer import wer, cer, process_words
from dotenv import load_dotenv

# ==========================================
# 1. CẤU HÌNH
# ==========================================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "lyrics_recognition")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tracks")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "dataset")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 2. TIỀN XỬ LÝ VĂN BẢN (Chuẩn hóa trước khi so khớp)
# ==========================================
def normalize_text(text):
    """
    Chuẩn hóa văn bản trước khi tính WER/CER:
    - Chuyển thường (lowercase)
    - Bỏ dấu câu và ký tự đặc biệt
    - Nén khoảng trắng thừa
    - Giữ nguyên dấu tiếng Việt (rất quan trọng)
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Loại bỏ các ký tự không phải chữ cái, số, khoảng trắng (giữ Unicode Vietnamese)
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    # Nén nhiều khoảng trắng thành một
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_lyrics_at_timerange(synced_lrc, start_sec=60, end_sec=90):
    """
    Trích xuất phần lời bài hát trong khoảng thời gian của snippet (mặc định 60s-90s)
    từ file LRC đồng bộ, thay vì so sánh với toàn bộ bài hát.

    Điều này giúp WER phản ánh chính xác hơn chất lượng nhận diện,
    vì Whisper chỉ nghe đoạn 30s chứ không phải cả bài.
    """
    if not synced_lrc:
        return None

    lines = synced_lrc.strip().split('\n')
    result_lines = []

    for line in lines:
        # Parse timestamp [MM:SS.xx]
        match = re.match(r'\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)', line)
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            total_seconds = minutes * 60 + seconds
            text = match.group(3).strip()

            if start_sec <= total_seconds <= end_sec and text:
                result_lines.append(text)

    return ' '.join(result_lines) if result_lines else None


def detect_language(text):
    """
    Phát hiện ngôn ngữ đơn giản dựa trên ký tự Unicode.
    Phân loại: 'vietnamese', 'korean', 'japanese', 'english_or_other'
    """
    if not text:
        return "unknown"

    # Đếm ký tự Vietnamese (có dấu)
    vn_chars = set('àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ')
    vn_count = sum(1 for c in text.lower() if c in vn_chars)

    # Đếm ký tự Hangul (Korean)
    kr_count = sum(1 for c in text if '\uac00' <= c <= '\ud7af' or '\u1100' <= c <= '\u11ff')

    # Đếm ký tự Japanese (Hiragana + Katakana)
    jp_count = sum(1 for c in text if '\u3040' <= c <= '\u30ff')

    total = len(text)
    if total == 0:
        return "unknown"

    if vn_count / total > 0.05:
        return "vietnamese"
    elif kr_count / total > 0.1:
        return "korean"
    elif jp_count / total > 0.1:
        return "japanese"
    else:
        return "english_or_other"


# ==========================================
# 3. TÍNH TOÁN WER/CER CHO TOÀN BỘ DATASET
# ==========================================
def evaluate_all_tracks():
    """
    Đánh giá WER/CER cho toàn bộ bài hát đã có predicted_text trong MongoDB.
    Trả về danh sách kết quả chi tiết và thống kê tổng hợp.
    """
    print("=" * 60)
    print("  MODULE B: ĐÁNH GIÁ WER/CER HỆ THỐNG")
    print("=" * 60)

    client = MongoClient(MONGO_URI)
    tracks_col = client[DB_NAME][COLLECTION_NAME]

    # Lấy tất cả bài có cả ground truth và predicted text
    query = {
        "lyrics.full_text": {"$exists": True, "$ne": None},
        "lyrics.predicted_text": {"$exists": True, "$ne": None}
    }
    tracks = list(tracks_col.find(query))
    print(f"\n[1/4] Tìm thấy {len(tracks)} bài hát có cả lời gốc và lời Whisper.")

    # ---- Tính WER/CER cho từng bài ----
    print("[2/4] Đang tính WER/CER cho từng bài...")
    results = []

    for track in tracks:
        track_id = track["_id"]
        title = track["metadata"]["title"]
        artists = ", ".join(track["metadata"]["artist"])
        predicted = track["lyrics"].get("predicted_text", "")
        full_text = track["lyrics"].get("full_text", "")
        synced_lrc = track["lyrics"].get("synced_lrc")
        start_time = track.get("snippet", {}).get("start_time", 60)
        end_time = track.get("snippet", {}).get("end_time", 90)

        # Ưu tiên dùng lời đồng bộ trong khoảng snippet thay vì toàn bộ bài
        ground_truth = extract_lyrics_at_timerange(synced_lrc, start_time, end_time)
        gt_source = "synced_lrc_segment"
        if not ground_truth:
            ground_truth = full_text
            gt_source = "full_text"

        # Chuẩn hóa
        ref = normalize_text(ground_truth)
        hyp = normalize_text(predicted)

        if not ref or not hyp:
            continue

        # Tính WER & CER
        try:
            word_error_rate = wer(ref, hyp)
            char_error_rate = cer(ref, hyp)

            # Phân tích lỗi chi tiết (substitutions, insertions, deletions)
            word_output = process_words(ref, hyp)
            n_sub = word_output.substitutions
            n_ins = word_output.insertions
            n_del = word_output.deletions
            n_hits = word_output.hits
            n_ref_words = n_sub + n_del + n_hits  # Tổng từ tham chiếu

        except Exception as e:
            print(f"  [Lỗi] {title}: {e}")
            continue

        # Phát hiện ngôn ngữ
        lang = detect_language(ground_truth)

        results.append({
            "track_id": track_id,
            "title": title,
            "artist": artists,
            "language": lang,
            "gt_source": gt_source,
            "wer": round(word_error_rate, 4),
            "cer": round(char_error_rate, 4),
            "substitutions": n_sub,
            "insertions": n_ins,
            "deletions": n_del,
            "hits": n_hits,
            "ref_words": n_ref_words,
            "ref_text_preview": ref[:100],
            "hyp_text_preview": hyp[:100]
        })

    print(f"  -> Đã tính xong cho {len(results)} bài hát.")

    # ---- Thống kê tổng hợp ----
    print("[3/4] Đang tổng hợp thống kê...")
    stats = compute_statistics(results)
    print_statistics(stats, results)

    # ---- Lưu kết quả ----
    print("[4/4] Đang lưu kết quả...")
    output_path = os.path.join(RESULTS_DIR, "wer_evaluation_results.json")
    output_data = {
        "total_evaluated": len(results),
        "statistics": stats,
        "per_track_results": results
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"  -> Đã lưu kết quả chi tiết vào: {output_path}")

    return results, stats


def compute_statistics(results):
    """Tính toán thống kê tổng hợp từ danh sách kết quả WER/CER."""
    if not results:
        return {}

    import numpy as np

    wer_values = [r["wer"] for r in results]
    cer_values = [r["cer"] for r in results]

    # Thống kê chung
    stats = {
        "overall": {
            "total_tracks": len(results),
            "wer_mean": round(float(np.mean(wer_values)), 4),
            "wer_median": round(float(np.median(wer_values)), 4),
            "wer_std": round(float(np.std(wer_values)), 4),
            "wer_min": round(float(np.min(wer_values)), 4),
            "wer_max": round(float(np.max(wer_values)), 4),
            "cer_mean": round(float(np.mean(cer_values)), 4),
            "cer_median": round(float(np.median(cer_values)), 4),
            "total_substitutions": sum(r["substitutions"] for r in results),
            "total_insertions": sum(r["insertions"] for r in results),
            "total_deletions": sum(r["deletions"] for r in results),
            "total_hits": sum(r["hits"] for r in results),
        },
        "by_language": {},
        "wer_distribution": {
            "excellent_0_20": 0,   # WER < 20%
            "good_20_40": 0,       # WER 20-40%
            "fair_40_60": 0,       # WER 40-60%
            "poor_60_80": 0,       # WER 60-80%
            "bad_80_plus": 0       # WER > 80%
        },
        "top_10_best": [],
        "top_10_worst": []
    }

    # Phân bố WER
    for r in results:
        w = r["wer"]
        if w < 0.2:
            stats["wer_distribution"]["excellent_0_20"] += 1
        elif w < 0.4:
            stats["wer_distribution"]["good_20_40"] += 1
        elif w < 0.6:
            stats["wer_distribution"]["fair_40_60"] += 1
        elif w < 0.8:
            stats["wer_distribution"]["poor_60_80"] += 1
        else:
            stats["wer_distribution"]["bad_80_plus"] += 1

    # Thống kê theo ngôn ngữ
    lang_groups = {}
    for r in results:
        lang = r["language"]
        if lang not in lang_groups:
            lang_groups[lang] = []
        lang_groups[lang].append(r)

    for lang, group in lang_groups.items():
        lang_wers = [r["wer"] for r in group]
        lang_cers = [r["cer"] for r in group]
        stats["by_language"][lang] = {
            "count": len(group),
            "wer_mean": round(float(np.mean(lang_wers)), 4),
            "wer_median": round(float(np.median(lang_wers)), 4),
            "cer_mean": round(float(np.mean(lang_cers)), 4),
        }

    # Top 10 best/worst WER
    sorted_results = sorted(results, key=lambda x: x["wer"])
    stats["top_10_best"] = [
        {"title": r["title"], "artist": r["artist"], "wer": r["wer"], "cer": r["cer"], "language": r["language"]}
        for r in sorted_results[:10]
    ]
    stats["top_10_worst"] = [
        {"title": r["title"], "artist": r["artist"], "wer": r["wer"], "cer": r["cer"], "language": r["language"]}
        for r in sorted_results[-10:]
    ]

    return stats


def print_statistics(stats, results):
    """In thống kê tổng hợp ra terminal."""
    overall = stats["overall"]

    print("\n" + "=" * 60)
    print("  KẾT QUẢ ĐÁNH GIÁ TỔNG HỢP")
    print("=" * 60)

    print(f"\n📊 Tổng số bài hát đánh giá: {overall['total_tracks']}")
    print(f"\n  WER (Word Error Rate):")
    print(f"    Trung bình (Mean):  {overall['wer_mean']:.2%}")
    print(f"    Trung vị (Median):  {overall['wer_median']:.2%}")
    print(f"    Độ lệch chuẩn:     {overall['wer_std']:.2%}")
    print(f"    Thấp nhất:         {overall['wer_min']:.2%}")
    print(f"    Cao nhất:          {overall['wer_max']:.2%}")
    print(f"\n  CER (Character Error Rate):")
    print(f"    Trung bình (Mean):  {overall['cer_mean']:.2%}")
    print(f"    Trung vị (Median):  {overall['cer_median']:.2%}")

    print(f"\n  Phân tích lỗi (Error Breakdown):")
    total_errors = overall['total_substitutions'] + overall['total_insertions'] + overall['total_deletions']
    if total_errors > 0:
        print(f"    Substitutions (thay thế sai): {overall['total_substitutions']:,} ({overall['total_substitutions']/total_errors:.1%})")
        print(f"    Insertions (thêm thừa):       {overall['total_insertions']:,} ({overall['total_insertions']/total_errors:.1%})")
        print(f"    Deletions (bỏ sót):           {overall['total_deletions']:,} ({overall['total_deletions']/total_errors:.1%})")
        print(f"    Correct hits (đúng):          {overall['total_hits']:,}")

    print(f"\n📈 Phân bố chất lượng WER:")
    dist = stats["wer_distribution"]
    total = overall['total_tracks']
    print(f"    Xuất sắc (< 20%):   {dist['excellent_0_20']:3d} bài ({dist['excellent_0_20']/total:.1%})")
    print(f"    Tốt (20-40%):       {dist['good_20_40']:3d} bài ({dist['good_20_40']/total:.1%})")
    print(f"    Trung bình (40-60%):{dist['fair_40_60']:3d} bài ({dist['fair_40_60']/total:.1%})")
    print(f"    Kém (60-80%):       {dist['poor_60_80']:3d} bài ({dist['poor_60_80']/total:.1%})")
    print(f"    Rất kém (> 80%):    {dist['bad_80_plus']:3d} bài ({dist['bad_80_plus']/total:.1%})")

    print(f"\n🌐 Thống kê theo ngôn ngữ:")
    for lang, data in stats["by_language"].items():
        print(f"    {lang:20s}: {data['count']:3d} bài | WER: {data['wer_mean']:.2%} | CER: {data['cer_mean']:.2%}")

    print(f"\n🏆 Top 5 bài WER tốt nhất:")
    for i, r in enumerate(stats["top_10_best"][:5], 1):
        print(f"    {i}. {r['title'][:40]:40s} WER: {r['wer']:.2%} ({r['language']})")

    print(f"\n⚠️  Top 5 bài WER kém nhất:")
    for i, r in enumerate(stats["top_10_worst"][:5], 1):
        print(f"    {i}. {r['title'][:40]:40s} WER: {r['wer']:.2%} ({r['language']})")

    print("\n" + "=" * 60)


# ==========================================
# 4. CHẠY CHÍNH
# ==========================================
if __name__ == "__main__":
    results, stats = evaluate_all_tracks()
