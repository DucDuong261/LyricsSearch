#!/usr/bin/env python3
"""One command from a song list to a working search index.

  python quickstart.py songs.txt

Steps: fetch lyrics (fetch_lyrics.py) -> build index (build_index.py).
Re-run any time after adding songs to songs.txt. Existing lyrics are kept;
add your own .txt files to my_lyrics/ for songs that were not found online.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script, *args):
    subprocess.run([sys.executable, str(HERE / script), *args], check=True)


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python quickstart.py songs.txt   (one 'Title - Artist' per line)")
    songs = sys.argv[1]
    run("fetch_lyrics.py", songs, "--out", "my_lyrics")
    run("build_index.py", "my_lyrics", "--out", "lyrics_data")
    print("\nReady. Try:  python scripts/search.py \"a few lyrics you remember\"")


if __name__ == "__main__":
    main()
