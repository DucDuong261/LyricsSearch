"""
pipeline/data_agent.py — AI Data Agent (Auto-Ingestion Orchestrator)
===================================================================
Agent tự động hóa hoàn toàn chu trình nạp dữ liệu nhạc Việt Nam mới nhất:
1. Tự động khám phá (Auto-Discovery) từ BXH YouTube Music Việt Nam (Top 100, Trending 20, Daily Hot)
   và các từ khóa tìm kiếm nhạc mới.
2. Tự động kiểm tra trùng lặp (Deduplication) với MongoDB.
3. Tự động lọc tiếng Việt (Vietnamese Language Filter).
4. Tự động đồng bộ Lời bài hát (Lyrics / Synced LRC qua LRCLIB).
5. Tự động tải đoạn Audio Snippet 30s chất lượng cao (16kHz Mono qua yt-dlp).
6. Tự động tạo Vector Embeddings (CLAP Audio & SBERT Lyrics) và đẩy vào Vector DB (Qdrant + FAISS).

Sử dụng:
  # Chạy 1 lần lấy 10 bài mới nhất:
  ./venv/bin/python pipeline/data_agent.py --limit 10

  # Chỉ chạy bước quét và tải (bỏ qua indexing vector):
  ./venv/bin/python pipeline/data_agent.py --limit 10 --skip-index

  # Chạy ở chế độ chạy thử nghiệm (không ghi vào DB):
  ./venv/bin/python pipeline/data_agent.py --dry-run
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import re
import time
import uuid
import argparse
import subprocess
import requests
from difflib import SequenceMatcher
from ytmusicapi import YTMusic

from config import (
    get_tracks_collection, get_qdrant_client,
    QDRANT_COLLECTION, CLAP_SAMPLE_RATE, SNIPPETS_DIR,
    FAISS_INDEX_PATH, ID_MAPPING_PATH, SBERT_MODEL
)
from pipeline.crawl import is_vietnamese_track

# ==========================================
# CẤU HÌNH & KHỞI TẠO
# ==========================================
tracks_col = get_tracks_collection()
ytmusic = YTMusic()

LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"
OUTPUT_DIR = str(SNIPPETS_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_title(title: str) -> str:
    """Làm sạch tiêu đề bài hát để tăng tỷ lệ khớp khi tra cứu API Lời."""
    title = re.sub(r'\(.*?Official.*?\)', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\[.*?Lyric.*?\]', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\(Video\)', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\(Audio\)', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\(MV\)', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\[MV\]', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s+-\s+.*', '', title)
    return title.strip()


def get_similarity(a: str, b: str) -> float:
    """Đo độ tương đồng giữa hai chuỗi văn bản."""
    return SequenceMatcher(None, (a or '').lower(), (b or '').lower()).ratio()


def get_ytdlp_bin() -> str:
    """Tự động định vị file thực thi yt-dlp trong virtualenv hoặc hệ thống."""
    import shutil
    venv_bin = Path(sys.executable).parent / "yt-dlp"
    if venv_bin.exists():
        return str(venv_bin)
    which_bin = shutil.which("yt-dlp")
    if which_bin:
        return which_bin
    return "yt-dlp"


class MusicDataAgent:
    """
    AI Data Agent chịu trách nhiệm thu thập, xử lý và nạp dữ liệu nhạc Việt tự động.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.stats = {
            "discovered": 0,
            "new_vietnamese": 0,
            "lyrics_synced": 0,
            "audio_downloaded": 0,
            "vectors_indexed": 0
        }

    # =========================================================================
    # BƯỚC 1: TỰ ĐỘNG KHÁM PHÁ BÀI HÁT VIỆT MỚI NHẤT
    # =========================================================================
    def discover_new_tracks(self, target_count: int = 10, source: str = "all", custom_query: str = None) -> list:
        """
        Quét các Bảng Xếp Hạng và tìm kiếm trên YouTube Music để tìm các bài hát mới.
        Chỉ giữ lại các bài Tiếng Việt và CHƯA CÓ trong MongoDB.
        Hỗ trợ custom_query để quét theo ca sĩ hoặc chủ đề yêu thích.
        """
        print(f"\n🔍 [Bước 1/4] AI Data Agent đang quét tìm các bài hát Việt mới nhất (Mục tiêu: {target_count} bài)...")
        candidates = []
        seen_video_ids = set()

        # 1.0 Quét theo truy vấn tùy chỉnh người dùng (nếu có)
        if custom_query:
            print(f"  -> 🎯 Đang ưu tiên quét theo từ khóa người dùng yêu cầu: '{custom_query}'...")
            try:
                results = ytmusic.search(custom_query, filter="songs", limit=max(50, target_count * 2))
                for track in results:
                    v_id = track.get('videoId')
                    if not v_id or v_id in seen_video_ids:
                        continue
                    seen_video_ids.add(v_id)

                    title = track.get('title', '')
                    artists = [a['name'] for a in track.get('artists', [])]

                    if not is_vietnamese_track(title, artists):
                        continue

                    if tracks_col.find_one({"_id": v_id}):
                        continue

                    candidates.append({
                        "video_id": v_id,
                        "title": title,
                        "artists": artists,
                        "release_year": track.get('year', 0) or 0
                    })

                    if len(candidates) >= target_count:
                        break
            except Exception as qe:
                print(f"     Lỗi tìm kiếm custom query '{custom_query}': {qe}")

        # 1.1 Quét từ YouTube Music Charts Việt Nam
        if len(candidates) < target_count and source in ["all", "charts"]:
            try:
                print("  -> Đang truy vấn YouTube Music Charts (Vietnam)...")
                charts = ytmusic.get_charts(country='VN')
                chart_videos = charts.get('videos', [])
                
                chart_playlist_ids = []
                if isinstance(chart_videos, list):
                    for item in chart_videos:
                        pid = item.get('playlistId')
                        if pid:
                            chart_playlist_ids.append((item.get('title', 'BXH'), pid))

                for p_title, p_id in chart_playlist_ids:
                    if len(candidates) >= target_count:
                        break
                    print(f"     * Đang duyệt playlist BXH: '{p_title}'...")
                    try:
                        pl = ytmusic.get_playlist(p_id, limit=100)
                        for track in pl.get('tracks', []):
                            v_id = track.get('videoId')
                            if not v_id or v_id in seen_video_ids:
                                continue
                            seen_video_ids.add(v_id)

                            title = track.get('title', '')
                            artists = [a['name'] for a in track.get('artists', [])]

                            # Lọc tiếng Việt
                            if not is_vietnamese_track(title, artists):
                                continue

                            # Kiểm tra MongoDB xem đã tồn tại chưa
                            if tracks_col.find_one({"_id": v_id}):
                                continue

                            candidates.append({
                                "video_id": v_id,
                                "title": title,
                                "artists": artists,
                                "release_year": track.get('year', 0) or 0
                            })

                            if len(candidates) >= target_count:
                                break
                    except Exception as pe:
                        print(f"       Lỗi duyệt playlist {p_id}: {pe}")

            except Exception as ce:
                print(f"  -> Không thể lấy charts: {ce}")

        # 1.2 Nếu chưa đủ số lượng, quét thêm qua kho từ khóa tìm kiếm phong phú
        if len(candidates) < target_count and source in ["all", "search"]:
            search_queries = [
                "nhạc việt mới nhất",
                "V-Pop thịnh hành",
                "bài hát mới phát hành",
                "nhạc trẻ hot tiktok việt nam",
                "Anh Trai Say Hi",
                "Anh Trai Vượt Ngàn Chông Gai",
                "nhạc indie việt nam hay nhất",
                "nhạc rap việt hay nhất",
                "nhạc ballad việt nam",
                "Sơn Tùng M-TP",
                "HIEUTHUHAI",
                "Vũ.",
                "Đen Vâu",
                "Soobin Hoàng Sơn",
                "Hoàng Dũng",
                "Grey D",
                "Dương Domic",
                "Phan Mạnh Quỳnh",
                "Wren Evans",
                "tlinh",
                "MCK",
                "buitruonglinh"
            ]
            for query in search_queries:
                if len(candidates) >= target_count:
                    break
                print(f"  -> Đang tìm kiếm từ khóa xu hướng: '{query}'...")
                try:
                    results = ytmusic.search(query, filter="songs", limit=50)
                    for track in results:
                        v_id = track.get('videoId')
                        if not v_id or v_id in seen_video_ids:
                            continue
                        seen_video_ids.add(v_id)

                        title = track.get('title', '')
                        artists = [a['name'] for a in track.get('artists', [])]

                        if not is_vietnamese_track(title, artists):
                            continue

                        if tracks_col.find_one({"_id": v_id}):
                            continue

                        candidates.append({
                            "video_id": v_id,
                            "title": title,
                            "artists": artists,
                            "release_year": track.get('year', 0) or 0
                        })

                        if len(candidates) >= target_count:
                            break
                except Exception as se:
                    print(f"     Lỗi tìm kiếm '{query}': {se}")

        self.stats["discovered"] = len(seen_video_ids)
        self.stats["new_vietnamese"] = len(candidates)

        print(f"  ✅ Đã phát hiện {len(candidates)} bài hát Việt Nam MỚI chưa có trong hệ thống.")
        for idx, item in enumerate(candidates, 1):
            print(f"     {idx}. {item['title']} - {', '.join(item['artists'])} (ID: {item['video_id']})")

        # Lưu metadata vào MongoDB
        if not self.dry_run and candidates:
            for item in candidates:
                doc = {
                    "metadata": {
                        "title": item["title"],
                        "artist": item["artists"],
                        "release_year": item["release_year"],
                        "video_id": item["video_id"],
                        "language": "vi"
                    },
                    "snippet": {
                        "start_time": None,
                        "end_time": None,
                        "path": None,
                        "is_processed": False
                    },
                    "lyrics": {
                        "full_text": None,
                        "synced_lrc": None
                    },
                    "local_assets": {
                        "status": "pending"
                    }
                }
                tracks_col.update_one({"_id": item["video_id"]}, {"$set": doc}, upsert=True)
            print(f"  💾 Đã lưu {len(candidates)} bài vào MongoDB (trạng thái: pending).")

        return candidates

    # =========================================================================
    # BƯỚC 2: TỰ ĐỘNG ĐỒNG BỘ LỜI BÀI HÁT (LYRICS SYNC)
    # =========================================================================
    def sync_lyrics(self, candidates: list = None) -> int:
        """
        Truy vấn API LRCLIB để lấy lời bài hát và lời đồng bộ thời gian (LRC).
        """
        print(f"\n📝 [Bước 2/4] AI Data Agent đang đồng bộ lời bài hát (Lyrics & LRC)...")
        
        if self.dry_run and candidates:
            pending_tracks = [
                {
                    "_id": c["video_id"],
                    "metadata": {"title": c["title"], "artist": c["artists"]}
                } for c in candidates
            ]
        elif candidates:
            candidate_ids = [c["video_id"] for c in candidates]
            pending_tracks = list(tracks_col.find({"_id": {"$in": candidate_ids}}))
        else:
            query = {"lyrics.full_text": None, "local_assets.status": "pending"}
            pending_tracks = list(tracks_col.find(query))

        if not pending_tracks:
            print("  -> Không có bài hát nào cần lấy lời.")
            return 0

        synced_count = 0

        for idx, track in enumerate(pending_tracks, 1):
            track_id = track["_id"]
            raw_title = track["metadata"]["title"]
            artists = track["metadata"].get("artist", [])
            primary_artist = artists[0] if artists else ""
            cleaned = clean_title(raw_title)

            print(f"  [{idx}/{len(pending_tracks)}] Tra cứu lời: {cleaned} - {primary_artist}...", end="", flush=True)

            try:
                params = {
                    "track_name": cleaned,
                    "artist_name": primary_artist
                }
                response = requests.get(LRCLIB_SEARCH_URL, params=params, timeout=10)
                matched_item = None

                if response.status_code == 200:
                    results = response.json()
                    for item in results:
                        api_track = item.get("trackName", "")
                        api_artist = item.get("artistName", "")
                        if get_similarity(cleaned, api_track) > 0.6 and get_similarity(primary_artist, api_artist) > 0.4:
                            matched_item = item
                            break

                if matched_item and matched_item.get("plainLyrics"):
                    plain_lyrics = matched_item.get("plainLyrics")
                    synced_lyrics = matched_item.get("syncedLyrics")

                    if not self.dry_run:
                        tracks_col.update_one(
                            {"_id": track_id},
                            {
                                "$set": {
                                    "lyrics.full_text": plain_lyrics,
                                    "lyrics.synced_lrc": synced_lyrics,
                                    "local_assets.status": "has_lyrics"
                                }
                            }
                        )
                    synced_count += 1
                    print(" [THÀNH CÔNG]")
                else:
                    if not self.dry_run:
                        tracks_col.update_one(
                            {"_id": track_id},
                            {"$set": {"local_assets.status": "lyrics_not_found"}}
                        )
                    print(" [CHƯA CÓ LỜI TRÊN LRCLIB]")

            except Exception as e:
                print(f" [LỖI: {e}]")

            time.sleep(0.3)

        self.stats["lyrics_synced"] = synced_count
        print(f"  ✅ Đã đồng bộ lời thành công cho {synced_count}/{len(pending_tracks)} bài hát.")
        return synced_count

    # =========================================================================
    # BƯỚC 3: TỰ ĐỘNG TẢI AUDIO SNIPPET 30 GIÂY
    # =========================================================================
    def download_snippets(self, candidates: list = None, start_time: str = "01:00", end_time: str = "01:30") -> int:
        """
        Tải trực tiếp đoạn 30s âm thanh qua yt-dlp và chuẩn hóa về 16kHz Mono WAV.
        Tải cho cả bài có lyrics hoặc chưa có lyrics để đảm bảo nhận diện audio được 100%.
        """
        print(f"\n🎵 [Bước 3/4] AI Data Agent đang tải đoạn âm thanh mẫu (Audio Snippet 30s)...")
        
        if self.dry_run and candidates:
            ready_tracks = [
                {
                    "_id": c["video_id"],
                    "metadata": {"title": c["title"], "artist": c["artists"]}
                } for c in candidates
            ]
        elif candidates:
            candidate_ids = [c["video_id"] for c in candidates]
            query = {
                "_id": {"$in": candidate_ids},
                "snippet.path": None
            }
            ready_tracks = list(tracks_col.find(query))
        else:
            query = {
                "local_assets.status": {"$in": ["has_lyrics", "lyrics_not_found", "pending", "audio_download_failed"]},
                "snippet.path": None
            }
            ready_tracks = list(tracks_col.find(query))

        if not ready_tracks:
            print("  -> Không có bài hát nào cần tải audio.")
            return 0

        downloaded_count = 0

        for idx, track in enumerate(ready_tracks, 1):
            v_id = track["_id"]
            title = track["metadata"]["title"]
            out_filename = f"{v_id}.wav"
            out_filepath = os.path.join(OUTPUT_DIR, out_filename)

            print(f"  [{idx}/{len(ready_tracks)}] Tải snippet: {title} (ID: {v_id})...", end="", flush=True)

            if self.dry_run:
                print(" [DRY-RUN: Mô phỏng tải 30s audio]")
                downloaded_count += 1
                continue

            if os.path.exists(out_filepath):
                print(" [ĐÃ TỒN TẠI TRÊN ĐĨA]")
                tracks_col.update_one(
                    {"_id": v_id},
                    {
                        "$set": {
                            "snippet.path": f"dataset/snippets/{out_filename}",
                            "snippet.is_processed": True,
                            "local_assets.status": "audio_ready"
                        }
                    }
                )
                downloaded_count += 1
                continue

            url = f"https://www.youtube.com/watch?v={v_id}"
            ytdlp_bin = get_ytdlp_bin()
            temp_raw_pattern = os.path.join(OUTPUT_DIR, f"temp_{v_id}.%(ext)s")

            try:
                # Bước 1: Tải file audio gốc bằng yt-dlp native downloader
                subprocess.run(
                    [ytdlp_bin, "--quiet", "--no-warnings", "-f", "bestaudio", "-o", temp_raw_pattern, url],
                    check=True, capture_output=True, text=True
                )

                # Tìm file tạm thực tế vừa tải (.webm, .m4a, v.v.)
                temp_files = list(Path(OUTPUT_DIR).glob(f"temp_{v_id}.*"))
                if not temp_files:
                    raise RuntimeError("Không tìm thấy file audio tạm vừa tải về")
                actual_temp = str(temp_files[0])

                # Bước 2: Dùng ffmpeg cắt 30s và chuyển chuẩn 16kHz Mono WAV (rất nhanh và không lỗi mạng)
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", start_time, "-t", "30",
                    "-i", actual_temp,
                    "-ar", "16000", "-ac", "1",
                    out_filepath
                ]
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)

                # Xóa file tạm
                if os.path.exists(actual_temp):
                    os.remove(actual_temp)

                tracks_col.update_one(
                    {"_id": v_id},
                    {
                        "$set": {
                            "snippet.path": f"dataset/snippets/{out_filename}",
                            "snippet.start_time": start_time,
                            "snippet.end_time": end_time,
                            "snippet.is_processed": True,
                            "local_assets.status": "audio_ready"
                        }
                    }
                )
                downloaded_count += 1
                print(" [THÀNH CÔNG]")
            except Exception as e:
                print(f" [LỖI: {e}]")
                for f in Path(OUTPUT_DIR).glob(f"temp_{v_id}.*"):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
                tracks_col.update_one(
                    {"_id": v_id},
                    {"$set": {"local_assets.status": "audio_download_failed"}}
                )

        self.stats["audio_downloaded"] = downloaded_count
        print(f"  ✅ Đã tải thành công {downloaded_count}/{len(ready_tracks)} đoạn audio mẫu.")
        return downloaded_count

    # =========================================================================
    # BƯỚC 4: TỰ ĐỘNG CẬP NHẬT VECTOR DATABASE (QDRANT & FAISS)
    # =========================================================================
    def update_vector_indices(self, candidate_ids: list = None) -> int:
        """
        Trích xuất vector embedding cho audio mới bằng CLAP và đẩy vào Qdrant.
        Đồng thời tái tạo lại FAISS index cho Lyrics.
        """
        print(f"\n🧠 [Bước 4/4] AI Data Agent đang nhúng Vector (CLAP Audio & SBERT Lyrics)...")

        if self.dry_run:
            print("  [DRY-RUN] Bỏ qua bước cập nhật Vector Database.")
            return 0

        # 4.1 Cập nhật Audio Embeddings vào Qdrant (Incremental Upsert)
        if candidate_ids:
            query = {"_id": {"$in": candidate_ids}, "local_assets.status": "audio_ready"}
        else:
            query = {"local_assets.status": "audio_ready", "snippet.path": {"$ne": None}}

        tracks_to_index = list(tracks_col.find(query))
        indexed_count = 0

        if tracks_to_index:
            try:
                import librosa
                import laion_clap
                from qdrant_client.models import PointStruct, VectorParams, Distance

                qdrant_client = get_qdrant_client()

                # Đảm bảo collection Qdrant tồn tại
                existing_cols = [c.name for c in qdrant_client.get_collections().collections]
                if QDRANT_COLLECTION not in existing_cols:
                    qdrant_client.create_collection(
                        collection_name=QDRANT_COLLECTION,
                        vectors_config=VectorParams(size=512, distance=Distance.COSINE)
                    )

                print(f"  -> Đang khởi tạo mô hình CLAP Audio Encoder để nhúng {len(tracks_to_index)} bài...")
                clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
                clap_model.load_ckpt()

                points = []
                for track in tracks_to_index:
                    v_id = track["_id"]
                    title = track["metadata"]["title"]
                    artists = track["metadata"].get("artist", [])
                    rel_path = track["snippet"]["path"]
                    full_path = os.path.join(str(Path(__file__).resolve().parent.parent), rel_path)

                    if not os.path.exists(full_path):
                        continue

                    try:
                        # Đọc và encode bằng CLAP
                        audio_data, _ = librosa.load(full_path, sr=CLAP_SAMPLE_RATE, mono=True)
                        audio_embed = clap_model.get_audio_embedding_from_data(x=[audio_data], use_tensor=False)[0]

                        # Dùng UUID xác định dựa trên video_id để upsert an toàn
                        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, v_id))
                        point = PointStruct(
                            id=point_id,
                            vector=audio_embed.tolist(),
                            payload={
                                "video_id": v_id,
                                "title": title,
                                "artist": ", ".join(artists),
                                "language": "vi",
                                "audio_path": rel_path
                            }
                        )
                        points.append(point)
                        indexed_count += 1
                    except Exception as ee:
                        print(f"     Lỗi encode audio bài {title}: {ee}")

                if points:
                    qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points)
                    for track in tracks_to_index:
                        tracks_col.update_one({"_id": track["_id"]}, {"$set": {"local_assets.status": "indexed"}})
                    print(f"  ✅ Đã upsert {len(points)} audio vectors mới vào Qdrant Collection '{QDRANT_COLLECTION}'.")
            except Exception as e:
                print(f"  ⚠️ Không thể nhúng CLAP Audio Vector: {e}")
        else:
            print("  -> Không có audio mới nào cần tạo index CLAP.")

        # 4.2 Cập nhật Lyrics Index (FAISS)
        # Chỉ chạy nếu có bài hát mới được đồng bộ lyrics
        if self.stats["lyrics_synced"] > 0:
            print("  -> Đang cập nhật lại FAISS Lyrics Index...")
            try:
                from sentence_transformers import SentenceTransformer
                import faiss
                import json

                sbert_model = SentenceTransformer(SBERT_MODEL)
                tracks = list(tracks_col.find({"lyrics.full_text": {"$exists": True}, "metadata.language": "vi"}))

                lyrics_chunks = []
                chunk_mapping = []

                for t in tracks:
                    text = t.get('lyrics', {}).get('full_text')
                    if isinstance(text, str) and text.strip():
                        raw_lines = text.split('\n')
                        lines = [line.strip() for line in raw_lines if len(line.strip()) > 5]
                        for i in range(len(lines)):
                            lyrics_chunks.append(lines[i])
                            chunk_mapping.append({"mongo_id": str(t['_id']), "matched_line": lines[i]})
                            if i < len(lines) - 1:
                                combined = lines[i] + " " + lines[i+1]
                                lyrics_chunks.append(combined)
                                chunk_mapping.append({"mongo_id": str(t['_id']), "matched_line": combined})

                if lyrics_chunks:
                    vectors = sbert_model.encode(lyrics_chunks, show_progress_bar=False, convert_to_numpy=True)
                    dimension = vectors.shape[1]
                    faiss.normalize_L2(vectors)

                    index = faiss.IndexFlatIP(dimension)
                    index.add(vectors)

                    faiss.write_index(index, str(FAISS_INDEX_PATH))
                    with open(str(ID_MAPPING_PATH), 'w', encoding='utf-8') as f:
                        json.dump(chunk_mapping, f, ensure_ascii=False, indent=2)

                    print(f"  ✅ Đã lưu FAISS Index với {index.ntotal} câu/đoạn lời bài hát.")
            except Exception as fe:
                print(f"  ⚠️ Lỗi khi cập nhật FAISS: {fe}")

        self.stats["vectors_indexed"] = indexed_count
        return indexed_count

    # =========================================================================
    # HÀM ĐIỀU PHỐI CHÍNH TOÀN DIỆN
    # =========================================================================
    def run_full_pipeline(self, limit: int = 10, source: str = "all", query: str = None, skip_index: bool = False):
        """
        Thực thi toàn bộ luồng: Khám phá -> Lọc -> Đồng bộ Lyrics -> Tải Audio -> Đẩy Vector DB.
        """
        start_time = time.time()
        print("=" * 70)
        print("🤖 AI DATA AGENT — BẮT ĐẦU QUY TRÌNH NẠP DỮ LIỆU TỰ ĐỘNG")
        print(f"Mục tiêu: {limit} bài | Nguồn: {source} | Từ khóa: {query or 'Tự động quét'} | Dry-run: {self.dry_run}")
        print("=" * 70)

        # 1. Khám phá
        candidates = self.discover_new_tracks(target_count=limit, source=source, custom_query=query)
        candidate_ids = [c["video_id"] for c in candidates]

        if not candidates:
            print("\n✨ Kho dữ liệu đã cập nhật các bài mới nhất! Không phát hiện thêm bài mới.")
            return

        # 2. Đồng bộ lời bài hát
        self.sync_lyrics(candidates=candidates)

        # 3. Tải đoạn âm thanh
        self.download_snippets(candidates=candidates)

        # 4. Tạo Vector & Cập nhật Index
        if not skip_index:
            self.update_vector_indices(candidate_ids=candidate_ids)
        else:
            print("\n⏩ Đã bỏ qua bước Vector Indexing theo yêu cầu (--skip-index).")

        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("📊 TỔNG KẾT BÁO CÁO CỦA AI DATA AGENT")
        print("=" * 70)
        print(f"  • Số bài quét được:           {self.stats['discovered']}")
        print(f"  • Bài hát Việt Nam mới thêm: {self.stats['new_vietnamese']}")
        print(f"  • Lời bài hát đã đồng bộ:    {self.stats['lyrics_synced']}")
        print(f"  • Đoạn âm thanh đã tải về:   {self.stats['audio_downloaded']}")
        print(f"  • Vector mới đã nạp vào DB:  {self.stats['vectors_indexed']}")
        print(f"  • Thời gian hoàn thành:       {elapsed:.1f} giây")
        print(f"  • Tổng bài hát trong DB:     {tracks_col.count_documents({})}")
        print("=" * 70)


# ==========================================
# GIAO DIỆN DÒNG LỆNH (CLI)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Data Agent: Tự động khám phá và nạp nhạc Việt mới nhất.")
    parser.add_argument("--limit", type=int, default=10, help="Số lượng bài hát mới tối đa cần nạp (mặc định: 10)")
    parser.add_argument("--query", type=str, default=None, help="Từ khóa / Tên ca sĩ cụ thể muốn Agent tìm kiếm (ví dụ: 'Anh Trai Say Hi', 'Sơn Tùng M-TP')")
    parser.add_argument("--source", type=str, choices=["all", "charts", "search"], default="all", help="Nguồn lấy bài hát (mặc định: all)")
    parser.add_argument("--skip-index", action="store_true", help="Bỏ qua bước nhúng vector embedding vào Qdrant/FAISS")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ tìm kiếm và hiển thị danh sách, không ghi dữ liệu")

    args = parser.parse_args()

    agent = MusicDataAgent(dry_run=args.dry_run)
    agent.run_full_pipeline(limit=args.limit, source=args.source, query=args.query, skip_index=args.skip_index)
