import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os
from pymongo import TEXT, ASCENDING
from config import get_tracks_collection, MONGO_URI, DB_NAME, COLLECTION_NAME, PROJECT_ROOT

def import_data():
    tracks_col = get_tracks_collection()
    
    # 1. Setup Indexes
    print("[1/3] Thiết lập indexes cho MongoDB...")
    tracks_col.create_index([
        ("metadata.title", TEXT),
        ("metadata.artist", TEXT)
    ], name="text_search_index", default_language="none", language_override="none")
    
    tracks_col.create_index([
        ("local_assets.status", ASCENDING)
    ], name="status_index")
    
    # 2. Check if already populated
    count = tracks_col.count_documents({})
    print(f"[2/3] Hiện tại database có {count} bản ghi.")
    
    dataset_file = str(PROJECT_ROOT / "dataset" / "dataset_processed.json")
    if not os.path.exists(dataset_file):
        print(f"Không tìm thấy file {dataset_file}!")
        return
        
    with open(dataset_file, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    print(f"[3/3] Đang nạp {len(records)} bài hát từ {dataset_file} vào MongoDB...")
    inserted_or_updated = 0
    for track in records:
        tracks_col.update_one(
            {"_id": track["_id"]},
            {"$set": track},
            upsert=True
        )
        inserted_or_updated += 1
        
    total = tracks_col.count_documents({})
    print(f"✅ Hoàn tất! Tổng số bài hát trong DB: {total}")

if __name__ == "__main__":
    import_data()
