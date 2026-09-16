import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from config import (
    get_tracks_collection, SBERT_MODEL,
    FAISS_INDEX_PATH, ID_MAPPING_PATH
)

print("Đang khởi động 'Bộ não' AI (Vui lòng đợi vài giây)...")

model = SentenceTransformer(SBERT_MODEL, device='cpu')
index = faiss.read_index(str(FAISS_INDEX_PATH))

with open(str(ID_MAPPING_PATH), 'r', encoding='utf-8') as f:
    chunk_mapping = json.load(f)

tracks_col = get_tracks_collection()

print("\n" + "="*50)
print("🚀 HỆ THỐNG TÌM KIẾM BÀI HÁT TỪNG CÂU ĐÃ SẴN SÀNG!")
print("="*50)

def search_song(query_text, top_k=3):
    # 1. BẮT BUỘC chuẩn hóa vector câu truy vấn
    query_vector = model.encode([query_text], normalize_embeddings=True)
    
    # 2. Tìm kiếm (đã bỏ top_k= theo lỗi cú pháp lúc nãy)
    distances, indices = index.search(np.array(query_vector).astype('float32'), 15)
    
    print("-" * 40)
    seen_ids = set()
    results_found = 0
    
    for i in range(15):
        if results_found >= top_k:
            break
            
        score = distances[0][i]
        match_index = indices[0][i]
        
        match_info = chunk_mapping[match_index]
        mongo_id = match_info["mongo_id"]
        matched_line = match_info["matched_line"]
        
        if mongo_id in seen_ids:
            continue
        seen_ids.add(mongo_id)
        
        track = tracks_col.find_one({"_id": mongo_id})
        
        if track:
            title = track['metadata']['title']
            artists = ", ".join(track['metadata']['artist'])
            results_found += 1
            print(f" [Top {results_found}] {title} - {artists}")
            print(f"         Câu khớp nhất: '{matched_line}'")
            # Đổi chữ "Độ lệch" thành "Độ chính xác" (Score max là 1.0)
            print(f"         Độ chính xác (Cosine): {score:.4f}")
            
    print("-" * 40)

if __name__ == "__main__":
    while True:
        user_input = input("\n🎵 Nhập câu hát bạn nhớ (hoặc gõ 'exit' để thoát): ").strip()
        
        if user_input.lower() == 'exit':
            print("\n👋 Tạm biệt!")
            break
            
        if not user_input:
            continue
            
        print(f"🔍 Đang quét hàng vạn câu hát: '{user_input}'...")
        search_song(user_input)