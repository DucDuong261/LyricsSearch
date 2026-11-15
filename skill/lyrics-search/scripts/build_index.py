#!/usr/bin/env python3
"""Build a local lyrics search index from a folder of lyrics files.

Input : a folder of .txt / .lrc files named "Title - Artist.txt"
        (or a .json file: [{"title":..., "artist":..., "lyrics":...}, ...])
Output: <out>/lyrics.index (FAISS) + <out>/songs.json (metadata + chunk map)

No database needed. Runs fully offline after the model is downloaded once.
"""
import argparse
import json
import re
import sys
from pathlib import Path

MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LRC_TS = re.compile(r"\[\d{1,2}:\d{2}(?:[.:]\d{1,3})?\]")
LRC_TAG = re.compile(r"^\[[a-zA-Z]+:.*\]$")


def clean_lines(text):
    lines = []
    for raw in text.splitlines():
        raw = raw.strip()
        if LRC_TAG.match(raw):
            continue
        raw = LRC_TS.sub("", raw).strip()
        if len(raw) > 5:
            lines.append(raw)
    return lines


def load_songs(src: Path):
    songs = []
    if src.is_file() and src.suffix == ".json":
        for s in json.loads(src.read_text(encoding="utf-8")):
            songs.append((s.get("title", "?"), s.get("artist", "?"), s["lyrics"]))
        return songs
    for f in sorted(src.rglob("*")):
        if f.suffix.lower() not in (".txt", ".lrc"):
            continue
        title, _, artist = f.stem.partition(" - ")
        songs.append((title.strip(), artist.strip() or "?", f.read_text(encoding="utf-8", errors="ignore")))
    return songs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="folder of lyrics files, or a .json file")
    ap.add_argument("--out", default="lyrics_data", help="output folder (default: lyrics_data)")
    args = ap.parse_args()

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    songs = load_songs(Path(args.source))
    if not songs:
        sys.exit("No lyrics found. Put .txt/.lrc files named 'Title - Artist.txt' in the folder.")

    meta, chunks, chunk_map = [], [], []
    for sid, (title, artist, text) in enumerate(songs):
        lines = clean_lines(text)
        if not lines:
            continue
        meta.append({"id": sid, "title": title, "artist": artist})
        # sliding window of 1, 2 and 3 consecutive lines
        for i in range(len(lines)):
            for n in (1, 2, 3):
                if i + n <= len(lines):
                    chunks.append(" ".join(lines[i:i + n]))
                    chunk_map.append({"song": sid, "line": chunks[-1]})

    print(f"{len(meta)} songs -> {len(chunks)} chunks. Embedding...")
    model = SentenceTransformer(MODEL, device="cpu")
    emb = model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(emb.shape[1])  # inner product == cosine (vectors normalized)
    index.add(np.asarray(emb, dtype="float32"))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "lyrics.index"))
    (out / "songs.json").write_text(
        json.dumps({"songs": meta, "chunks": chunk_map}, ensure_ascii=False), encoding="utf-8")
    print(f"Done. Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
