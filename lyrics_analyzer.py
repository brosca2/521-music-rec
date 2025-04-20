import requests
import string
# import logging # removed logging import
from typing import Set # keep set for type hinting
import os
import pickle

# configure logging # removed logging configuration
# logging.basicConfig(level=logging.info, format='%(asctime)s - %(levelname)s - %(message)s')

# --- stopword fetching ---
# fetch stopwords when the module loads (assuming success)
STOPWORDS = set() # initialize as an empty set
# try: # removed try/except block
# logging.info("fetching stopwords list...") # removed log
# use a reliable gist url for stopwords
stopwords_url = "https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt"
stopwords_list_response = requests.get(stopwords_url, timeout=10) # added timeout
stopwords_list_response.raise_for_status() # raise an exception for bad status codes
stopwords_list = stopwords_list_response.content
STOPWORDS = set(stopwords_list.decode('utf-8').splitlines()) # specify utf-8 decoding
# logging.info(f"successfully fetched and loaded {len(stopwords)} stopwords.") # removed log
# except requests.exceptions.timeout: # removed exception handling
    # logging.error("timeout occurred while fetching stopwords. using an empty set.") # removed log
    # stopwords remains empty
# except requests.exceptions.requestexception as e: # removed exception handling
    # logging.error(f"error fetching stopwords: {e}. using an empty set as fallback.") # removed log
    # stopwords remains empty
# except exception as e: # removed exception handling
    # logging.error(f"an unexpected error occurred during stopword fetching: {e}. using an empty set.") # removed log
    # stopwords remains empty

# --- core processing function ---
def process_lyrics_for_word_set(lyrics: str) -> Set[str]:
    """
    process lyrics into a set of unique, lowercase, non-stop words without punctuation.

    args:
        lyrics (str): the raw lyrics text.

    returns:
        set[str]: a set of processed words.
    """
    if not lyrics or not lyrics.strip():
        # logging.debug("received empty or whitespace-only lyrics.") # removed log
        return set()

    # try: # removed try/except block
    # remove punctuation using translate
    # create translation table for punctuation removal
    translator = str.maketrans('', '', string.punctuation)
    no_punct = lyrics.translate(translator)

    # tokenize (simple split) and convert to lowercase
    words = no_punct.lower().split()

    # filter out stopwords and short tokens (like single letters)
    # use the globally loaded stopwords set
    processed_words = {word for word in words if word not in STOPWORDS and len(word) > 1}

    # logging.debug(f"processed lyrics into {len(processed_words)} unique non-stop words.") # removed log
    return processed_words

    # except exception as e: # removed try/except block
        # logging.error(f"error processing lyrics: {e}\nlyrics snippet: {lyrics[:100]}...") # removed log
        # return set() # return empty set on error

# --- analyzer class ---
class LyricsAnalyzer:
    """
    analyze lyrics to extract significant words for similarity comparison
    using word sets.
    """
    def __init__(self):
        """
        initialize the lyricsanalyzer.
        (stopwords are loaded globally)
        """
        # log initialization and stopword status # removed logging
        # if stopwords:
            # logging.info(f"lyricsanalyzer initialized for word set analysis with {len(stopwords)} stopwords.")
        # else:
            # logging.warning("lyricsanalyzer initialized for word set analysis without stopwords (fetch failed).")
        pass # constructor does nothing now

    def analyze_lyrics(self, lyrics: str, song_dir_path: str) -> Set[str]:
        """
        analyze lyrics, using caching if available. returns a set of processed words.

        checks for a cached result in song_dir_path/lyrics_analysis.pkl.
        if found, loads it. otherwise, processes lyrics and saves the result.

        args:
            lyrics (str): the lyrics text.
            song_dir_path (str): the path to the directory containing the song's files.

        returns:
            set[str]: a set of unique, lowercase, non-stop words.
        """
        cache_filename = "lyrics_analysis.pkl"
        cache_path = os.path.join(song_dir_path, cache_filename)

        # check for cache
        if os.path.exists(cache_path):
            # try: # removed try/except block
            with open(cache_path, 'rb') as f:
                # logging.debug(f"loading cached lyric analysis from: {cache_path}") # removed log
                result_set = pickle.load(f)
                # logging.info(f"successfully loaded cached lyric analysis for: {os.path.basename(song_dir_path)}") # removed log
                return result_set
            # except (filenotfounderror, pickle.unpicklingerror, eoferror, ioerror) as e: # removed exception handling
                # logging.warning(f"failed to load cache file {cache_path}: {e}. re-analyzing.") # removed log
                # proceed to analysis if cache loading fails

        # if cache doesn't exist or loading failed, perform analysis
        # logging.debug(f"analyzing lyrics for: {os.path.basename(song_dir_path)} (cache not found or failed to load)") # removed log
        # logging.debug(f"analyzing lyrics starting with: {lyrics[:50]}...") # removed log
        result_set = process_lyrics_for_word_set(lyrics)
        # logging.debug(f"analysis complete. found {len(result_set)} words.") # removed log

        # save the result to cache
        # try: # removed try/except block
        # ensure the directory exists
        os.makedirs(song_dir_path, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(result_set, f)
        # logging.info(f"saved lyric analysis cache to: {cache_path}") # removed log
        # except (ioerror, pickle.picklingerror) as e: # removed exception handling
            # logging.error(f"failed to save cache file {cache_path}: {e}") # removed log

        return result_set