---
name: lyrics-search
description: Finds a song's title and artist from a few remembered lyrics or a recorded clip, by semantic search over the user's own lyrics collection (built for Vietnamese music, works in any language). Use when the user asks "what song has the lyrics...", "tìm bài hát có câu...", or gives an audio clip to identify.
---

# Lyrics Search

Finds songs from lyrics you half-remember. It matches by **meaning**, not exact words, so a misheard or partial line still works.

No database, no API keys, no accounts. Everything runs on your machine.

**Network use:** the only network access is the one-time download of the open models (`paraphrase-multilingual-MiniLM-L12-v2`, and `openai/whisper-small` for `--audio`) from Hugging Face. Your lyrics and queries are never uploaded. The scripts read no environment variables and contain no credentials.

## How it works (in one minute)

1. **You provide lyrics** (a folder of text files, one per song). A tiny sample is in `examples/lyrics/`.
2. `build_index.py` turns every 1-3 line window into a vector and saves a local index. Do this once, and again whenever you add songs.
3. `search.py` embeds your query the same way and returns the closest songs with a confidence score.
4. With `--audio`, Whisper transcribes the clip first, then the same search runs.

## Setup (once)

```bash
pip install -r requirements.txt
```
The first run downloads the models (~500 MB for text; ~1 GB more if you use `--audio`).

## Usage

**1. Put lyrics in a folder**, one file per song, named `Title - Artist.txt` (`.lrc` works too, timestamps are stripped):
```
my_lyrics/
  Chiều mưa phố vắng - Ca Sĩ Mẫu.txt
```
Or use one JSON file: `[{"title": "...", "artist": "...", "lyrics": "..."}]`.

**2. Build the index:**
```bash
python scripts/build_index.py my_lyrics --out lyrics_data
```

**3. Search:**
```bash
python scripts/search.py "đi một mình dưới mưa nhớ lời hứa của em"
python scripts/search.py --audio my_recording.wav
```

## Reading the result

```json
{
  "query": "...",
  "confident": true,
  "results": [{"title": "...", "artist": "...", "matched_line": "...", "score": 0.82}]
}
```
- `score` is cosine similarity (0 to 1). **≥ 0.60** = likely right, **below** = probably not in your collection.
- `confident: false` means: tell the user no good match was found, then show the closest candidates as guesses. Never present them as certain.
- `matched_line` is the part of the song that matched. Show it so the user can verify.

## Instructions for the agent

- If `lyrics_data/` does not exist, ask the user for their lyrics folder and run `build_index.py` first.
- Pass the user's lyrics verbatim as the query; do not translate or "fix" them.
- For audio, prefer clips of 5-30 seconds with a clear voice. Vietnamese is the default; use `--lang en` etc. for other languages.
- Only return title, artist and the matched line. Do not print full copyrighted lyrics.
- Only index lyrics the user has the right to use.

## Limits

- It only finds songs that are **in your collection**. It does not search the internet.
- Whisper can mis-hear sung vocals; if audio search fails, ask the user to type the lyrics.
- Accuracy drops for very short queries (1-2 words).
