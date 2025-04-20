#!/usr/bin/env python3
"""script to generate language.txt files by detecting languages in lyrics."""

import os
import glob
import argparse
from langdetect import detect # removed LangDetectException as try/except is removed
# import logging # removed logging import

# configure logging # removed logging configuration
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_languages(lyrics_text, chunk_size=1000):
    """detect languages in lyrics, analyzing chunks for multilingual content."""
    if not lyrics_text or not lyrics_text.strip():
        # logging.debug("empty lyrics text provided for language detection.") # removed log
        return []

    languages = set()

    # first, detect language of the entire text (assuming detection succeeds)
    # try: # removed try/except
    main_lang = detect(lyrics_text)
    languages.add(main_lang)
    # except LangDetectException as e: # removed try/except
        # logging.warning(f"language detection failed for full text: {e}") # removed log

    # if text is long, analyze chunks for multiple languages
    if len(lyrics_text) > chunk_size:
        chunks = [lyrics_text[i:i+chunk_size] for i in range(0, len(lyrics_text), chunk_size)]
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # skip empty chunks
                # try: # removed try/except
                chunk_lang = detect(chunk)
                languages.add(chunk_lang)
                # except LangDetectException as e: # removed try/except
                    # logging.debug(f"language detection failed for chunk {i}: {e}") # removed log

    return list(languages)

def generate_language_files(songs_dir):
    """create language.txt files for each song based on detected languages."""
    # find all lyrics files
    lyrics_files = glob.glob(os.path.join(songs_dir, '*', 'lyrics.txt'))

    if not lyrics_files:
        # logging.warning(f"no lyrics.txt files found in subdirectories of {songs_dir}") # removed log
        return

    # logging.info(f"found {len(lyrics_files)} lyrics files to process") # removed log

    # process each lyrics file
    for file_path in lyrics_files:
        song_dir = os.path.dirname(file_path)
        song_name = os.path.basename(song_dir)
        language_file_path = os.path.join(song_dir, 'language.txt')

        # try: # removed try/except block
        # read lyrics
        with open(file_path, 'r', encoding='utf-8') as f:
            lyrics = f.read()

        # basic check for empty or placeholder lyrics
        if not lyrics or not lyrics.strip() or "lyrics not found" in lyrics.lower():
            # logging.warning(f"skipping empty or placeholder lyrics for song: {song_name}") # removed log
            continue

        # detect languages
        languages = detect_languages(lyrics)

        if not languages:
            # logging.warning(f"no languages detected for '{song_name}', skipping language file creation") # removed log
            continue

        # write language file
        with open(language_file_path, 'w', encoding='utf-8') as f:
            # write one language code per line
            for lang in languages:
                f.write(f"{lang}\n")

        # logging.info(f"created language file for '{song_name}' with languages: {languages}") # removed log

        # except Exception as e: # removed try/except block
            # logging.error(f"error processing lyrics for '{song_name}': {e}") # removed log

def main():
    """parse arguments and run the language file generator."""
    parser = argparse.ArgumentParser(description='generate language.txt files for songs')
    parser.add_argument('--songs_dir', required=True, help='path to the directory containing song subdirectories')

    args = parser.parse_args()

    # validate songs directory # removed validation check
    # if not os.path.isdir(args.songs_dir):
        # logging.error(f"songs directory not found: {args.songs_dir}") # removed log
        # return 1 # removed exit

    # generate language files
    generate_language_files(args.songs_dir)

    return 0

if __name__ == '__main__':
    exit(main())