import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from config import get_tracks_collection, SBERT_MODEL, FAISS_INDEX_PATH, ID_MAPPING_PATH

# 1. Kết nối MongoDB (Singleton từ config.py)
tracks_col = get_tracks_collection()

# 2. Khởi tạo SBERT
print("Đang load model SBERT...")
model = SentenceTransformer(SBERT_MODEL)

# 3. Lấy dữ liệu
print("Đang lấy dữ liệu từ MongoDB...")
query = {"lyrics.full_text": {"$exists": True}}
tracks = list(tracks_col.find(query))

# 4. CHUNKING: Tách bài hát thành từng câu
print("Đang phân rã bài hát bằng kỹ thuật Sliding Window...")
lyrics_chunks = []
chunk_mapping = []

for t in tracks:
    text = t.get('lyrics', {}).get('full_text')
    
    if isinstance(text, str) and text.strip():
        # Lọc sạch các dòng trống và ngắn trước
        raw_lines = text.split('\n')
        lines = [line.strip() for line in raw_lines if len(line.strip()) > 5]
        
        for i in range(len(lines)):
            # 1. Thêm Vector Dòng đơn
            lyrics_chunks.append(lines[i])
            chunk_mapping.append({
                "mongo_id": str(t['_id']),
                "matched_line": lines[i]
            })
            
            # 2. Thêm Vector 2 dòng liên tiếp
            if i < len(lines) - 1:
                combined_2_lines = lines[i] + " " + lines[i+1]
                lyrics_chunks.append(combined_2_lines)
                chunk_mapping.append({
                    "mongo_id": str(t['_id']),
                    "matched_line": combined_2_lines
                })
                
            # 3. Thêm Vector 3 dòng liên tiếp
            if i < len(lines) - 2:
                combined_3_lines = lines[i] + " " + lines[i+1] + " " + lines[i+2]
                lyrics_chunks.append(combined_3_lines)
                chunk_mapping.append({
                    "mongo_id": str(t['_id']),
                    "matched_line": combined_3_lines
                })

# Đã dời 2 dòng print lên đây để báo cáo trước khi nhúng
print(f"Kỹ thuật mới đã băm {len(tracks)} bài hát thành {len(lyrics_chunks)} vector!")
print("Bắt đầu nhúng Vector (Sẽ tốn thời gian hơn trước)...")

# Nhúng Vector kèm Normalize
embeddings = model.encode(lyrics_chunks, normalize_embeddings=True)

# ĐỔI IndexFlatL2 thành IndexFlatIP (Inner Product = Cosine Similarity)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension) 
index.add(np.array(embeddings).astype('float32'))

# Lưu lại
faiss.write_index(index, str(FAISS_INDEX_PATH))
with open(str(ID_MAPPING_PATH), 'w', encoding='utf-8') as f:
    json.dump(chunk_mapping, f, ensure_ascii=False)

print("Đã tạo xong bộ não Indexing mới chuẩn Cosine!")