import os
import re
import sys
from dotenv import load_dotenv
import lyricsgenius

def clean_lyrics(lyrics_text):
    """remove headers, annotations, and extra whitespace from lyrics."""
    # note: this function assumes input is valid text or none
    if not lyrics_text:
        return ""
    cleaned = re.sub(r'\[.*?\]\n?', '', lyrics_text) # remove bracketed annotations
    lines = cleaned.split('\n')
    meaningful_lines = [
        line for line in lines
        if line.strip() and not line.strip().endswith('Lyrics') and "contributor" not in line.lower()
        and len(line.strip()) > 2
    ]

    cleaned = "\n".join(meaningful_lines).strip()
    cleaned = re.sub(r'\d*Embed$', '', cleaned).strip()

    return cleaned

def fetch_and_save_lyrics():
    """fetch and save lyrics for songs in the songs directory."""

    # load environment variables, assuming .env exists
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    # if not os.path.exists(env_path): # removed existence check
    #     load_dotenv()
    # else:
    load_dotenv(dotenv_path=env_path) # always try loading from specified path

    genius_token = os.getenv('GENIUS_ACCESS_TOKEN')
    # if not genius_token: # removed token check
    #     sys.exit(1)

    # initialize genius api client, assuming token is valid
    genius = lyricsgenius.Genius(genius_token, verbose=False, remove_section_headers=True, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"])

    songs_base_dir = os.path.join(os.path.dirname(__file__), 'songs')
    # if not os.path.isdir(songs_base_dir): # removed directory check
    #     sys.exit(1)

    song_directories = [d for d in os.listdir(songs_base_dir)
                        if os.path.isdir(os.path.join(songs_base_dir, d))]

    if not song_directories:
        # print(f"no song subdirectories found in {songs_base_dir}.") # removed print
        return # exit if no directories found

    print(f"found {len(song_directories)} song directories. starting lyric fetch...")

    for song_title_raw in song_directories:
        song_dir_path = os.path.join(songs_base_dir, song_title_raw)
        lyrics_file_path = os.path.join(song_dir_path, 'lyrics.txt')

        print(f"\nprocessing: {song_title_raw}")

        # removed outer try...except block
        song_search_result = genius.search_song(song_title_raw)

        if song_search_result and hasattr(song_search_result, 'lyrics'):
            lyrics_raw = song_search_result.lyrics
            print(f"  > found lyrics for '{song_search_result.title}' by {song_search_result.artist}")

            lyrics_cleaned = clean_lyrics(lyrics_raw)

            if not lyrics_cleaned:
                 # print(f"  > warning: lyrics found but became empty after cleaning for '{song_title_raw}'. skipping save.") # removed warning print
                 continue # skip if cleaning results in empty string

            # removed inner try...except block for file writing
            with open(lyrics_file_path, 'w', encoding='utf-8') as f:
                f.write(lyrics_cleaned)
            print(f"  > successfully cleaned and saved lyrics to {lyrics_file_path}")
            # except ioerror as e: # removed ioerror handling
            #     print(f"  > error writing lyrics file {lyrics_file_path}: {e}", file=sys.stderr)

        # else: # removed handling for lyrics not found
            # print(f"  > could not find lyrics for '{song_title_raw}' on genius.")

        # except exception as e: # removed outer exception handling
        #     print(f"  > error processing '{song_title_raw}': {e}", file=sys.stderr)


    print("\nlyric fetching process completed.")

if __name__ == "__main__":
    fetch_and_save_lyrics()