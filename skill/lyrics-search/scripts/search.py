#!/usr/bin/env python3
"""Find a song from a few remembered lyrics (or a sung/recorded clip).

  python search.py "lời bài hát nhớ mang máng"
  python search.py --audio clip.wav        # transcribes with Whisper, then searches

Prints JSON: [{"title","artist","matched_line","score"}, ...]
"""
import argparse
import json
import sys
from pathlib import Path

MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
WHISPER = "openai/whisper-small"
DEFAULT_THRESHOLD = 0.60  # below this, the match is probably wrong


def transcribe(path, lang):
    import torch
    from transformers import pipeline
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    asr = pipeline("automatic-speech-recognition", model=WHISPER, device=dev,
                   generate_kwargs={"task": "transcribe", "language": lang})
    return asr(path)["text"].strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="?", help="lyrics text to search")
    ap.add_argument("--audio", help="audio file (wav/mp3) to transcribe and search")
    ap.add_argument("--data", default="lyrics_data", help="folder made by build_index.py")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--lang", default="vi", help="Whisper language (default: vi)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = ap.parse_args()

    if not args.query and not args.audio:
        ap.error("give lyrics text or --audio")
    data = Path(args.data)
    if not (data / "lyrics.index").exists():
        sys.exit(f"No index in '{data}'. Run: python build_index.py <lyrics_folder> --out {data}")

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    query = args.query or transcribe(args.audio, args.lang)
    db = json.loads((data / "songs.json").read_text(encoding="utf-8"))
    songs, chunks = db["songs"], db["chunks"]
    by_id = {s["id"]: s for s in songs}

    index = faiss.read_index(str(data / "lyrics.index"))
    vec = SentenceTransformer(MODEL, device="cpu").encode([query], normalize_embeddings=True)
    k = min(40, index.ntotal)
    scores, ids = index.search(np.asarray(vec, dtype="float32"), k)

    results, seen = [], set()
    for score, i in zip(scores[0], ids[0]):
        c = chunks[i]
        if c["song"] in seen:
            continue
        seen.add(c["song"])
        s = by_id[c["song"]]
        results.append({"title": s["title"], "artist": s["artist"],
                        "matched_line": c["line"], "score": round(float(score), 3)})
        if len(results) >= args.top:
            break

    out = {"query": query, "confident": bool(results and results[0]["score"] >= args.threshold),
           "results": results}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
