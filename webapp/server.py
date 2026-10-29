"""
webapp/server.py — Backend cho giao diện web AI Music Matcher (SPA)
====================================================================
Thay thế lớp giao diện Streamlit bằng một API nhẹ (Starlette) + frontend tĩnh
(webapp/static). Toàn bộ logic AI tái sử dụng các engine sẵn có:
  - hybrid_engine.search_hybrid        — Hybrid Fusion (Lời hát + Audio CLAP/Qdrant)
  - pipeline.data_agent.MusicDataAgent — Thu thập nhạc mới
  - audio_analysis                     — Phân tích tín hiệu
  - Whisper + SBERT + FAISS            — Tìm kiếm qua lời hát

Chạy:  python -m uvicorn webapp.server:app --port 8000
"""

import base64
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    get_tracks_collection, DATASET_DIR,
    FAISS_INDEX_PATH, ID_MAPPING_PATH,
    WHISPER_MODEL, WHISPER_LANGUAGE, SBERT_MODEL,
    MIN_AUDIO_DURATION, LYRICS_SEARCH_THRESHOLD
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
BASE_DIR = str(ROOT)

# ==========================================
# 1. LOAD AI (nạp nền khi server khởi động)
# ==========================================
_state = {"ready": False, "error": None, "stage": "Đang khởi động..."}
_models = {}
# Whisper/FAISS không an toàn khi gọi song song → tuần tự hóa các lượt suy luận
_infer_lock = threading.Lock()


def _load_all_systems():
    try:
        import faiss
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import pipeline

        _state["stage"] = "Đang nạp Whisper..."
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _models["whisper"] = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL,
            device=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            generate_kwargs={"task": "transcribe", "language": WHISPER_LANGUAGE}
        )
        _state["stage"] = "Đang nạp SBERT & FAISS..."
        _models["sbert"] = SentenceTransformer(SBERT_MODEL, device="cpu")
        _load_lyrics_index()
        _models["tracks"] = get_tracks_collection()
        _state["ready"] = True
        _state["stage"] = "Sẵn sàng"
    except Exception as e:  # pragma: no cover - hiển thị lỗi lên UI
        _state["error"] = str(e)
        _state["stage"] = "Lỗi khi nạp mô hình"


def _load_lyrics_index():
    """Nạp (hoặc nạp lại sau khi AI Agent cập nhật) FAISS index lời bài hát."""
    import faiss
    _models["index"] = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(str(ID_MAPPING_PATH), "r", encoding="utf-8") as f:
        _models["mapping"] = json.load(f)


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=_load_all_systems, daemon=True).start()
    yield


# ==========================================
# 2. TIỆN ÍCH
# ==========================================
def _resolve_path(raw_path):
    if not isinstance(raw_path, str) or not raw_path.strip():
        return ""
    safe_path = raw_path.replace("\\", "/")
    if not os.path.isabs(safe_path):
        safe_path = os.path.join(BASE_DIR, safe_path)
    return safe_path


def _audio_url(path):
    """Chuyển đường dẫn file trong dataset/ thành URL /media/... (None nếu ngoài dataset)."""
    if not path or not os.path.exists(path):
        return None
    try:
        rel = Path(path).resolve().relative_to(DATASET_DIR.resolve())
    except ValueError:
        return None
    return "/media/" + rel.as_posix()


def _fig_to_data_url(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# Biểu đồ matplotlib theo tông tối để khớp giao diện
_DARK_RC = {
    "figure.facecolor": "#181818", "axes.facecolor": "#181818",
    "savefig.facecolor": "#181818", "axes.edgecolor": "#3e3e3e",
    "axes.labelcolor": "#b3b3b3", "text.color": "#ffffff",
    "xtick.color": "#b3b3b3", "ytick.color": "#b3b3b3",
    "grid.color": "#3e3e3e", "legend.facecolor": "#242424",
    "legend.edgecolor": "#3e3e3e", "font.family": "DejaVu Sans",
}


def _plot(fn, *args, **kwargs):
    with plt.rc_context(_DARK_RC):
        return _fig_to_data_url(fn(*args, **kwargs))


def _serialize(item):
    """Bỏ các trường numpy nặng (ref_entry) và thêm audio_url."""
    out = {}
    for k, v in item.items():
        if k in ("ref_entry", "audio_path"):
            continue
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        out[k] = v
    out["audio_url"] = _audio_url(item.get("audio_path", ""))
    return out


def _save_upload(data, suffix=".wav"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name


def _clean(msg):
    """Bỏ biểu tượng cảnh báo ở đầu thông báo của engine (UI tự hiển thị icon)."""
    return msg.replace("⚠️", "").strip()


def _not_ready():
    msg = _state["error"] or "Mô hình AI đang được nạp, vui lòng đợi trong giây lát."
    return JSONResponse({"error": msg}, status_code=503)


# ==========================================
# 3. LOGIC TÌM KIẾM (tương đương demoweb.py)
# ==========================================
def search_from_text(query_text, top_k=3):
    sbert, index = _models["sbert"], _models["index"]
    mapping, tracks_col = _models["mapping"], _models["tracks"]
    query_vector = sbert.encode([query_text], normalize_embeddings=True)
    fetch_k = min(40, index.ntotal)
    distances, indices = index.search(np.array(query_vector).astype("float32"), fetch_k)
    seen_ids, seen_songs, results = set(), set(), []
    for i in range(fetch_k):
        if len(results) >= top_k:
            break
        match_info = mapping[indices[0][i]]
        mongo_id = match_info["mongo_id"]
        if mongo_id in seen_ids:
            continue
        seen_ids.add(mongo_id)
        track = tracks_col.find_one({"_id": mongo_id})
        if not track:
            continue
        title = track["metadata"]["title"]
        artist = track["metadata"]["artist"]
        artist_str = ", ".join(artist) if isinstance(artist, list) else str(artist)
        sig = (title.strip().lower(), artist_str.strip().lower())
        if sig in seen_songs:
            continue
        seen_songs.add(sig)
        results.append({
            "video_id": mongo_id,
            "title": title,
            "artist": artist_str,
            "matched_line": match_info["matched_line"],
            "score": float(distances[0][i]),
            "audio_path": _resolve_path(track.get("snippet", {}).get("path")),
            "full_lyrics": track.get("lyrics", {}).get("full_text", "")
        })
    return results


def _lyrics_payload(results, transcription=None):
    threshold = LYRICS_SEARCH_THRESHOLD
    payload = {"mode": "lyrics", "transcription": transcription, "threshold": threshold}
    if not results:
        payload["error"] = "Rất tiếc, hệ thống không trả về kết quả nào!"
        return payload
    best = results[0]["score"]
    if best < threshold:
        payload["error"] = (
            f"Không tìm thấy bài hát phù hợp. Độ tin cậy cao nhất chỉ đạt {best:.2f}, "
            f"thấp hơn ngưỡng {threshold:.2f}."
        )
        return payload
    keep = [results[0]] + [r for r in results[1:] if r["score"] >= threshold - 0.1]
    payload["results"] = [_serialize(r) for r in keep]
    return payload


def _run_audio_search(mode, audio_path):
    from audio_encoder import validate_audio_duration

    if mode == "hybrid":
        from hybrid_engine import search_hybrid
        results, trans, msg = search_hybrid(
            audio_path, _models["whisper"], _models["sbert"], _models["index"],
            _models["mapping"], _models["tracks"], top_k=3, min_duration=MIN_AUDIO_DURATION
        )
        if msg.startswith("⚠️"):
            return {"mode": mode, "error": _clean(msg)}
        if not results:
            return {"mode": mode, "error": "Không tìm thấy bài hát phù hợp!"}
        clean_trans = trans if trans and trans != "Nhiễu/Không nghe rõ" and not trans.startswith("Lỗi") else None
        return {"mode": mode, "transcription": clean_trans, "elapsed": msg,
                "results": [_serialize(r) for r in results]}

    # mode == "lyrics"
    import librosa
    is_valid, _, msg = validate_audio_duration(audio_path, min_duration=MIN_AUDIO_DURATION)
    if not is_valid:
        return {"mode": mode, "error": _clean(msg)}
    audio_array, sr = librosa.load(audio_path, sr=16000)
    text = _models["whisper"]({"array": audio_array, "sampling_rate": sr})["text"].strip()
    if not text or len(text) < 5:
        return {"mode": mode, "error": "AI không nghe rõ lời bài hát hoặc đoạn thu âm quá ồn. Vui lòng thử lại!"}
    return _lyrics_payload(search_from_text(text), transcription=text)


def _run_agent(limit, source, query):
    """Chạy AI Data Agent, gom log in ra để hiển thị trên giao diện."""
    from pipeline.data_agent import MusicDataAgent
    col = get_tracks_collection()
    before = col.count_documents({})
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            MusicDataAgent(dry_run=False).run_full_pipeline(limit=limit, source=source, query=query)
    except Exception as e:
        return {"error": f"AI Data Agent gặp lỗi: {e}", "log": log.getvalue()[-6000:]}
    if _state["ready"]:
        _load_lyrics_index()
    after = col.count_documents({})
    return {"before": before, "after": after, "added": after - before, "log": log.getvalue()[-6000:]}


# ==========================================
# 4. API ENDPOINTS
# ==========================================
async def index_page(request):
    return FileResponse(STATIC_DIR / "index.html")


async def status(request):
    return JSONResponse({**_state, "min_duration": MIN_AUDIO_DURATION})


async def search_text(request):
    if not _state["ready"]:
        return _not_ready()
    body = await request.json()
    query = (body.get("query") or "").strip()
    if len(query) < 3:
        return JSONResponse({"error": "Vui lòng nhập lời bài hát dài hơn một chút!"}, status_code=400)

    def work():
        with _infer_lock:
            return _lyrics_payload(search_from_text(query))
    return JSONResponse(await run_in_threadpool(work))


async def search_audio(request):
    mode = request.path_params["mode"]
    if mode not in ("hybrid", "lyrics"):
        return JSONResponse({"error": "Chế độ không hợp lệ"}, status_code=404)
    if not _state["ready"]:
        return _not_ready()
    form = await request.form()
    upload = form.get("audio")
    if upload is None:
        return JSONResponse({"error": "Thiếu file âm thanh"}, status_code=400)
    suffix = os.path.splitext(upload.filename or "")[1].lower() or ".wav"
    tmp_path = _save_upload(await upload.read(), suffix=suffix)

    def work():
        try:
            with _infer_lock:
                return _run_audio_search(mode, tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return JSONResponse(await run_in_threadpool(work))


async def run_agent(request):
    body = await request.json()
    limit = max(3, min(50, int(body.get("limit") or 10)))
    source = body.get("source") if body.get("source") in ("all", "charts", "search") else "all"
    query = (body.get("query") or "").strip() or None

    def work():
        with _infer_lock:
            return _run_agent(limit, source, query)
    return JSONResponse(await run_in_threadpool(work))


async def list_tracks(request):
    q = (request.query_params.get("q") or "").strip()
    filt = {"snippet.path": {"$ne": None}}
    if q:
        import re
        rx = {"$regex": re.escape(q), "$options": "i"}
        filt["$or"] = [{"metadata.title": rx}, {"metadata.artist": rx}]

    def work():
        col = get_tracks_collection()
        cur = col.find(filt, {"metadata.title": 1, "metadata.artist": 1, "snippet.path": 1}).limit(40)
        return [{
            "id": str(t["_id"]),
            "title": t["metadata"]["title"],
            "artist": ", ".join(t["metadata"]["artist"]) if isinstance(t["metadata"]["artist"], list) else str(t["metadata"]["artist"]),
        } for t in cur]
    return JSONResponse({"tracks": await run_in_threadpool(work)})


def _analyze_file(path, title):
    from audio_analysis import extract_audio_features, plot_full_analysis, get_feature_summary
    features = extract_audio_features(path)
    return {
        "title": title,
        "summary": get_feature_summary(features),
        "plot": _plot(plot_full_analysis, features, title=title),
    }


async def analyze(request):
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/"):
        form = await request.form()
        upload = form.get("audio")
        if upload is None:
            return JSONResponse({"error": "Thiếu file âm thanh"}, status_code=400)
        tmp_path = _save_upload(await upload.read(), suffix=os.path.splitext(upload.filename or "")[1] or ".wav")

        def work():
            try:
                return _analyze_file(tmp_path, upload.filename or "File tải lên")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        return JSONResponse(await run_in_threadpool(work))

    body = await request.json()
    track_id = body.get("track_id")

    def work():
        track = get_tracks_collection().find_one({"_id": track_id})
        if not track:
            return {"error": "Không tìm thấy bài hát"}
        path = _resolve_path(track.get("snippet", {}).get("path"))
        if not os.path.exists(path):
            return {"error": f"Không tìm thấy file audio: {path}"}
        artist = track["metadata"]["artist"]
        title = f"{track['metadata']['title']} - {', '.join(artist) if isinstance(artist, list) else artist}"
        result = _analyze_file(path, title)
        result["audio_url"] = _audio_url(path)
        return result
    return JSONResponse(await run_in_threadpool(work))


async def wer_stats(request):
    path = DATASET_DIR / "wer_evaluation_results.json"
    if not path.exists():
        return JSONResponse({"error": "Chưa có kết quả đánh giá. Chạy `python evaluation.py` trước."}, status_code=404)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse({
        "statistics": data["statistics"],
        "wer_values": [r["wer"] for r in data["per_track_results"]],
    })


app = Starlette(
    routes=[
        Route("/", index_page),
        Route("/api/status", status),
        Route("/api/search/text", search_text, methods=["POST"]),
        Route("/api/search/audio/{mode}", search_audio, methods=["POST"]),
        Route("/api/agent", run_agent, methods=["POST"]),
        Route("/api/tracks", list_tracks),
        Route("/api/analyze", analyze, methods=["POST"]),
        Route("/api/wer", wer_stats),
        Mount("/media", app=StaticFiles(directory=str(DATASET_DIR)), name="media"),
        Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
    ],
    lifespan=lifespan,
)
