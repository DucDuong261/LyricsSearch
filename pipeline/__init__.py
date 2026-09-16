"""
Package: pipeline
=================
Các script thu thập và tiền xử lý dữ liệu (chạy tuần tự, 1 lần).

Thứ tự chạy:
  1. python pipeline/crawl.py              — Crawl playlist từ YouTube Music
  2. python pipeline/lyrics_sync.py        — Đồng bộ lời bài hát từ LRCLIB
  3. python pipeline/download_snippets.py  — Tải & cắt 30s audio snippet
  4. python pipeline/whisper_transcribe.py — Cho Whisper nghe và nhận diện lời
  5. python pipeline/import_mongo.py       — Nạp backup JSON vào MongoDB
  6. python pipeline/export_song_list.py   — Xuất danh sách bài hát ra file text
"""
