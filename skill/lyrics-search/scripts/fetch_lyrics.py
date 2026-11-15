#!/usr/bin/env python3
"""Download lyrics for a list of songs from LRCLIB (free, public, no API key).

  python fetch_lyrics.py songs.txt --out my_lyrics
  python fetch_lyrics.py --song "Tên bài" --artist "Ca sĩ" --out my_lyrics

songs.txt: one song per line, "Title - Artist" (artist optional).
Saves "Title - Artist.txt" files that build_index.py can read directly.
Songs LRCLIB does not have are listed at the end so you can paste them by hand.

Network: only https://lrclib.net/api/search is contacted, with the song title/artist.
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

API = "https://lrclib.net/api/search"
LRC_TS = re.compile(r"\[\d{1,2}:\d{2}(?:[.:]\d{1,3})?\]")


def sim(a, b):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def search(title, artist):
    params = {"track_name": title}
    if artist:
        params["artist_name"] = artist
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "lyrics-search-skill/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def best_lyrics(title, artist):
    """Return (lyrics_text, matched_title, matched_artist) or None."""
    try:
        hits = search(title, artist)
    except Exception as e:  # network error, rate limit...
        print(f"   network error: {e}")
        return None
    best, best_score = None, 0.0
    for h in hits:
        text = h.get("plainLyrics") or LRC_TS.sub("", h.get("syncedLyrics") or "")
        if not text.strip() or h.get("instrumental"):
            continue
        score = sim(title, h.get("trackName", ""))
        if artist:
            score = 0.7 * score + 0.3 * sim(artist, h.get("artistName", ""))
        if score > best_score:
            best, best_score = (text, h.get("trackName", title), h.get("artistName", artist)), score
    return best if best_score >= 0.6 else None


def safe(name):
    return re.sub(r'[\\/:*?"<>|]', "", name).strip()[:80]


def parse_line(line):
    title, _, artist = line.partition(" - ")
    return title.strip(), artist.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("songs_file", nargs="?", help="text file, one 'Title - Artist' per line")
    ap.add_argument("--song")
    ap.add_argument("--artist", default="")
    ap.add_argument("--out", default="my_lyrics")
    args = ap.parse_args()

    if args.song:
        wanted = [(args.song, args.artist)]
    elif args.songs_file:
        lines = Path(args.songs_file).read_text(encoding="utf-8").splitlines()
        wanted = [parse_line(l) for l in lines if l.strip() and not l.startswith("#")]
    else:
        ap.error("give a songs file or --song")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    missing = []
    for n, (title, artist) in enumerate(wanted, 1):
        print(f"[{n}/{len(wanted)}] {title} - {artist}")
        found = best_lyrics(title, artist)
        if not found:
            missing.append(f"{title} - {artist}".strip(" -"))
            print("   not found")
        else:
            text, t, a = found
            (out / f"{safe(t)} - {safe(a or artist or '?')}.txt").write_text(text, encoding="utf-8")
            print(f"   saved ({t} / {a})")
        time.sleep(1)  # be polite to the free API

    print(f"\nSaved {len(wanted) - len(missing)}/{len(wanted)} songs to {out.resolve()}")
    if missing:
        print("Not found (paste their lyrics into a .txt file in that folder):")
        for m in missing:
            print("  -", m)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
