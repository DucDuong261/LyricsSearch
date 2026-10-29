import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from config import (
    get_tracks_collection, SBERT_MODEL,
    FAISS_INDEX_PATH, ID_MAPPING_PATH
)

# ==========================================
# 1. TẢI CÁC THÀNH PHẦN CỐT LÕI (Chỉ chạy 1 lần)
# ==========================================
print("Đang khởi động 'Bộ não' AI (Vui lòng đợi vài giây)...")

model = SentenceTransformer(SBERT_MODEL, device='cpu')
index = faiss.read_index(str(FAISS_INDEX_PATH))

with open(str(ID_MAPPING_PATH), 'r', encoding='utf-8') as f:
    id_mapping = json.load(f)

tracks_col = get_tracks_collection()

print("\n" + "="*50)
print("🚀 HỆ THỐNG TÌM KIẾM BÀI HÁT ĐÃ SẴN SÀNG!")
print("="*50)

# ==========================================
# 2. HÀM TÌM KIẾM CHÍNH
# ==========================================
def search_song(query_text, top_k=3):
    # Nhúng câu hát thành vector kèm chuẩn hóa Cosine
    query_vector = model.encode([query_text], normalize_embeddings=True)
    
    # Tìm kiếm trong FAISS (lấy dư để lọc trùng bài hát)
    fetch_count = min(top_k * 5, index.ntotal)
    distances, indices = index.search(np.array(query_vector).astype('float32'), fetch_count)
    
    print("-" * 35)
    seen_ids = set()
    found = 0
    for i in range(fetch_count):
        if found >= top_k:
            break
        score = distances[0][i]
        match_index = indices[0][i]
        
        match_info = id_mapping[match_index]
        mongo_id = match_info["mongo_id"] if isinstance(match_info, dict) else match_info
        matched_line = match_info.get("matched_line", "") if isinstance(match_info, dict) else ""
        
        if mongo_id in seen_ids:
            continue
        seen_ids.add(mongo_id)
        
        track = tracks_col.find_one({"_id": mongo_id})
        
        if track:
            found += 1
            title = track['metadata']['title']
            artists = ", ".join(track['metadata']['artist'])
            print(f" [Top {found}] {title} - {artists}")
            if matched_line:
                print(f"         Câu khớp: \"{matched_line}\"")
            print(f"         Độ tương đồng (Cosine): {score:.4f}")
        else:
            print(f" [Lỗi] Không tìm thấy dữ liệu cho ID: {mongo_id}")
    print("-" * 35)

# ==========================================
# 3. VÒNG LẶP TƯƠNG TÁC (CLI)
# ==========================================
if __name__ == "__main__":
    while True:
        # Nhận đầu vào từ bàn phím
        user_input = input("\n🎵 Nhập câu hát bạn nhớ (hoặc gõ 'exit' để thoát): ").strip()
        
        # Kiểm tra lệnh thoát
        if user_input.lower() == 'exit':
            print("\n👋 Tạm biệt! Chúc hệ thống của nhóm em chạy tốt nhé!")
            break
            
        # Bỏ qua nếu người dùng lỡ bấm Enter mà chưa nhập gì
        if not user_input:
            continue
            
        # Thực hiện tìm kiếm
        print(f"🔍 Đang phân tích ngữ nghĩa: '{user_input}'...")
        search_song(user_input)