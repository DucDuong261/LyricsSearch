import os
import time
import torch
import librosa
from transformers import pipeline
from pymongo import MongoClient
from dotenv import load_dotenv

# ==========================================
# 1. CẤU HÌNH CƠ SỞ DỮ LIỆU
# ==========================================
load_dotenv()
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
tracks_col = client[os.getenv("DB_NAME", "lyrics_recognition")][os.getenv("COLLECTION_NAME", "tracks")]

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH WHISPER
# ==========================================
print("Đang khởi động AI (Whisper)...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Hệ thống đang chạy trên: {device.upper()}")

# Đã tháo bỏ 'language': 'vietnamese' để AI tự động phát hiện ngôn ngữ
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=device,
    generate_kwargs={"task": "transcribe"} 
)

# ==========================================
# 3. LUỒNG TEST TỐC ĐỘ (5 BÀI)
# ==========================================
def test_whisper_speed():
    # Sử dụng .limit(5) để chỉ quét 5 bản ghi đầu tiên
    query = {"local_assets.status": "audio_ready"}
    test_tracks = list(tracks_col.find(query).limit(5))
    
    print(f"\n=== BẮT ĐẦU BÀI TEST SỨC MẠNH CPU CHO {len(test_tracks)} BÀI HÁT ===")
    
    for index, track in enumerate(test_tracks):
        track_id = track["_id"]
        title = track["metadata"]["title"]
        audio_path = track["snippet"]["path"]
        
        print(f"\n[{index+1}/5] Đang nghe: {title}")
        
        # Bắt đầu bấm giờ
        start_time = time.time()
        
        try:
            # Xử lý âm thanh
            audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
            result = transcriber({"array": audio_array, "sampling_rate": sampling_rate})
            predicted_text = result["text"].strip()
            
            # Kết thúc bấm giờ
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Cập nhật kết quả vào MongoDB
            tracks_col.update_one(
                {"_id": track_id},
                {
                    "$set": {
                        "lyrics.predicted_text": predicted_text,
                        "local_assets.status": "whisper_done"
                    }
                }
            )
            
            print(f"   -> Thời gian CPU xử lý: {processing_time:.2f} giây")
            print(f"   -> Kết quả: {predicted_text[:80]}...")
            
        except Exception as e:
            print(f"   -> Lỗi khi xử lý file: {e}")

if __name__ == "__main__":
    test_whisper_speed()