"""
Package: pipeline
=================
Các script thu thập và tiền xử lý dữ liệu (chạy tuần tự, 1 lần).

Thứ tự chạy:
  1. python pipeline/crawl.py              — Crawl playlist từ YouTube Music (chỉ lọc Tiếng Việt)
  2. python pipeline/lyrics_sync.py        — Đồng bộ lời bài hát từ LRCLIB
  3. python pipeline/download_snippets.py  — Tải & cắt 30s audio snippet
  4. python pipeline/whisper_transcribe.py — Cho Whisper nghe và nhận diện lời (language=vi)
  5. python build_index_from_mongo.py      — Nhúng lyrics → FAISS Vector Index
  6. python pipeline/build_audio_index.py  — Nhúng audio → CLAP → Qdrant Vector Index
  7. python pipeline/import_mongo.py       — Nạp backup JSON vào MongoDB
  8. python pipeline/export_song_list.py   — Xuất danh sách bài hát ra file text
"""
