import faiss
import json
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. TẢI CÁC THÀNH PHẦN CỐT LÕI (Chỉ chạy 1 lần)
# ==========================================
print("Đang khởi động 'Bộ não' AI (Vui lòng đợi vài giây)...")

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
index = faiss.read_index("lyrics_db.index")

with open('id_mapping.json', 'r') as f:
    id_mapping = json.load(f)

client = MongoClient("mongodb://localhost:27017/")
tracks_col = client["lyrics_recognition"]["tracks"]

print("\n" + "="*50)
print("🚀 HỆ THỐNG TÌM KIẾM BÀI HÁT ĐÃ SẴN SÀNG!")
print("="*50)

# ==========================================
# 2. HÀM TÌM KIẾM CHÍNH
# ==========================================
def search_song(query_text, top_k=3):
    # Nhúng câu hát thành vector
    query_vector = model.encode([query_text])
    
    # Tìm kiếm trong FAISS
    distances, indices = index.search(np.array(query_vector).astype('float32'), top_k)
    
    print("-" * 30)
    for i in range(top_k):
        score = distances[0][i]
        match_index = indices[0][i]
        
        mongo_id = id_mapping[match_index]
        
        # Đã sửa lỗi ObjectId: Truy vấn bằng chuỗi (string) trực tiếp
        track = tracks_col.find_one({"_id": mongo_id})
        
        if track:
            title = track['metadata']['title']
            artists = ", ".join(track['metadata']['artist'])
            print(f" [Top {i+1}] {title} - {artists}")
            print(f"         (Độ lệch: {score:.2f})")
        else:
            print(f" [Top {i+1}] [Lỗi] Không tìm thấy dữ liệu cho ID: {mongo_id}")
    print("-" * 30)

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