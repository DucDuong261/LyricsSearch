import os
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
print("Đang tải mô hình Whisper (Lần đầu sẽ hơi lâu vì phải tải vài GB data)...")

# Xác định phần cứng: Dùng GPU (cuda) nếu có, không thì dùng CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Hệ thống đang chạy AI trên: {device.upper()}")

# Khởi tạo Pipeline nhận diện giọng nói
# Sử dụng bản 'small' để cân bằng giữa tốc độ và độ chính xác
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=device,
    generate_kwargs={ "task": "transcribe"} 
)

# ==========================================
# 3. LUỒNG XỬ LÝ CHÍNH
# ==========================================
def transcribe_audio_dataset():
    # Chỉ lấy các bài đã có file audio sẵn sàng và chưa được AI nghe
    query = {"local_assets.status": "audio_ready"}
    ready_tracks = list(tracks_col.find(query))
    
    print(f"\n=== BẮT ĐẦU CHO AI NGHE {len(ready_tracks)} BÀI HÁT ===")
    
    success_count = 0
    
    for index, track in enumerate(ready_tracks):
        track_id = track["_id"]
        title = track["metadata"]["title"]
        audio_path = track["snippet"]["path"]
        
        print(f"[{index+1}/{len(ready_tracks)}] Đang nhận diện: {title}...")
        
        try:
            # Load file audio bằng librosa (đảm bảo đúng tần số 16kHz cho Whisper)
            audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
            
            # Đẩy vào AI
            result = transcriber({"array": audio_array, "sampling_rate": sampling_rate})
            predicted_text = result["text"].strip()
            
            # Lưu kết quả vào DB
            tracks_col.update_one(
                {"_id": track_id},
                {
                    "$set": {
                        "lyrics.predicted_text": predicted_text,
                        "local_assets.status": "whisper_done" # Đánh dấu AI đã xử lý xong
                    }
                }
            )
            print(f"   -> Kết quả: {predicted_text[:50]}...") # In ra 50 ký tự đầu cho vui mắt
            success_count += 1
            
        except Exception as e:
            print(f"   -> Lỗi khi xử lý file: {e}")

    print(f"\n=== HOÀN TẤT NHẬN DIỆN {success_count} BÀI HÁT ===")

if __name__ == "__main__":
    transcribe_audio_dataset()