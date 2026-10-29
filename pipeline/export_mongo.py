"""
pipeline/export_mongo.py — Xuất toàn bộ MongoDB ra dataset/dataset_processed.json
==================================================================================
Ngược lại với import_mongo.py: dùng để chia sẻ kho dữ liệu (metadata, lời bài hát,
synced lyrics, thông tin snippet) qua GitHub. Người khác chỉ cần chạy
`python pipeline/import_mongo.py` là có dữ liệu giống hệt.

Bản ghi đã có trong file JSON nhưng không còn trong MongoDB vẫn được giữ lại.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from config import get_tracks_collection, PROJECT_ROOT

DATASET_FILE = PROJECT_ROOT / "dataset" / "dataset_processed.json"


def export_data():
    records = {}
    if DATASET_FILE.exists():
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            records = {t["_id"]: t for t in json.load(f)}
    before = len(records)

    for track in get_tracks_collection().find({}):
        # Chuẩn hóa đường dẫn kiểu Windows (dataset/snippets\x.wav) để chạy được trên mọi OS
        path = track.get("snippet", {}).get("path")
        if isinstance(path, str):
            track["snippet"]["path"] = path.replace("\\", "/")
        records[track["_id"]] = track

    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        # Giữ thứ tự cũ + indent=4 như file gốc để diff trên GitHub gọn
        json.dump(list(records.values()), f, ensure_ascii=False, indent=4, default=str)

    print(f"Đã xuất {len(records)} bài hát (trước đó {before}) → {DATASET_FILE}")


if __name__ == "__main__":
    export_data()
