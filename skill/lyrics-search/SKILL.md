---
name: lyrics-search
description: Finds a song's title and artist from a few remembered lyrics or a recorded clip, by meaning-based search over a lyrics collection that the skill builds for you from a list of song names (built for Vietnamese music, works in any language). Use when the user asks "what song has the lyrics...", "tìm bài hát có câu...", "tên bài này là gì", or gives an audio clip to identify.
---

# Lyrics Search

Finds songs from lyrics you half-remember. It matches by **meaning**, not exact words, so a misheard or partial line still works.

You start from a blank page: **no database, no API keys, no account, and you do not need to know how to code.** Tell the agent which songs you care about; it downloads the lyrics, builds the search, and answers.

## For the user (non-developer): what to say

Just talk to your agent, for example:

- "Set up lyrics-search. My songs are: Em Của Ngày Hôm Qua - Sơn Tùng M-TP, Hãy Trao Cho Anh - Sơn Tùng M-TP."
- "Which song has the line 'đi một mình dưới mưa nhớ lời hứa'?"
- "What song is this?" (attach a recording)
- "Add these songs too: ..."

The agent does everything below. You never need to open a terminal.

## Instructions for the agent

### First-time setup (blank page)
1. Install dependencies once: `pip install -r requirements.txt` (in this skill folder). First model download is large (~500 MB text, ~1 GB more for audio); tell the user it is a one-time wait.
2. Ask the user which songs they want to be able to recognise, as `Title - Artist`. If they have no list, offer to start with a few artists or songs they name. Write them to `songs.txt`, one per line.
3. Run `python scripts/quickstart.py songs.txt`. This fetches lyrics, then builds the index into `lyrics_data/`.
4. Read the "Not found" list from the output. For those songs, ask the user to paste the lyrics (or an `.lrc`/`.txt` file), save each as `my_lyrics/Title - Artist.txt`, then run `python scripts/build_index.py my_lyrics --out lyrics_data` again.

### Searching
- Text: `python scripts/search.py "lyrics as the user typed them"`
- Audio: `python scripts/search.py --audio clip.wav` (wav/mp3/flac, 5-30 s, clear voice; no ffmpeg needed). Default language is Vietnamese; use `--lang en` etc.
- Pass the user's words verbatim. Do not translate or "correct" them.

### Adding songs later
Append to `songs.txt` and re-run `quickstart.py`, or drop `Title - Artist.txt` files into `my_lyrics/` and re-run `build_index.py`.

### Answering
Result format:
```json
{"query": "...", "confident": true,
 "results": [{"title": "...", "artist": "...", "matched_line": "...", "score": 0.82}]}
```
- `score` is cosine similarity. **≥ 0.60** is likely right.
- `confident: false`: say no good match was found in the collection and offer the closest as guesses, never as certain. Suggest adding the song if the user thinks it should be there.
- For audio, `confident` is always false: present the top 3 as candidates and show what was heard (`query`) so the user can judge a mis-transcription.
- Show title, artist and the matched line only. Do not print full lyrics.

## Where the lyrics come from

- **Automatic:** [LRCLIB](https://lrclib.net), a free community lyrics database with a public API (no key). Coverage of popular Vietnamese songs is good; rare or very new songs may be missing.
- **Manual:** anything the user pastes or provides as `.txt`/`.lrc` files. Only index lyrics you have the right to use; the index stays on your machine.

## Privacy and network

The only network calls are: (1) the one-time download of the open models (`paraphrase-multilingual-MiniLM-L12-v2`, and `openai/whisper-small` for audio) from Hugging Face; (2) `fetch_lyrics.py` sending song titles and artists to `lrclib.net`. Audio files and search queries never leave the machine. The scripts read no environment variables and contain no credentials.

## Manual use (developers)

```bash
pip install -r requirements.txt
python scripts/fetch_lyrics.py examples/songs.txt --out my_lyrics   # or use your own .txt files
python scripts/build_index.py my_lyrics --out lyrics_data
python scripts/search.py "câu lời nhớ mang máng"
```
Lyrics files are named `Title - Artist.txt`; a JSON file `[{"title","artist","lyrics"}]` also works.

## Limits

- It only finds songs that are **in your collection**; it does not search the whole internet.
- Audio search is approximate: in our test on real 30 s Vietnamese song clips (40-song collection), the right song was ranked first in 3 of 6 clips. Whisper mis-hears sung vocals, so audio results are always returned as candidates (`confident: false`) with a `note`. Show the top 3 and what was heard, and ask the user to type the lyrics if none fit.
- Very short queries (1-2 words) are unreliable.
