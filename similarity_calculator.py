import os
import glob
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# import logging # removed logging import
import csv
from langdetect import detect # removed LangDetectException as try/except is removed
# import yake # no longer needed for fallback
from lyrics_analyzer import LyricsAnalyzer # import the new lyricsanalyzer

# configure logging # removed logging configuration
# logging.basicconfig(level=logging.info, format='%(asctime)s - %(levelname)s - %(message)s')


# --- helper functions ---

def _get_song_name_from_path(filepath):
    """extract the song directory name from a filepath."""
    # try: # removed try/except block
    parts = filepath.split(os.sep)
    # handle potential valueerror if 'songs' is not in path
    if 'songs' not in parts:
        # logging.warning(f"could not extract song name from path (missing 'songs'): {filepath}") # removed log
        return None
    songs_index = parts.index('songs')
    if songs_index + 1 < len(parts):
        return parts[songs_index + 1]
    else:
        # logging.warning(f"could not extract song name from path (no directory after 'songs'): {filepath}") # removed log
        return None
    # except (valueerror, indexerror): # removed try/except block
        # logging.warning(f"could not extract song name from path: {filepath}") # removed log
        # return none

# --- language detection and reading ---

def read_language_files(song_dir_path):
    """read language codes from language.txt files in song directories."""
    language_file_path = os.path.join(song_dir_path, 'language.txt')

    if not os.path.exists(language_file_path):
        # logging.debug(f"no language.txt file found for song: {os.path.basename(song_dir_path)}") # removed log
        return []

    languages = []
    # try: # removed try/except block
    with open(language_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # skip empty lines and comments
            line = line.strip()
            if line and not line.startswith('#'):
                languages.append(line)

    # if languages: # removed conditional logging
        # logging.debug(f"read languages from file for '{os.path.basename(song_dir_path)}': {languages}")
    # else:
        # logging.debug(f"no languages found in language.txt for '{os.path.basename(song_dir_path)}'")

    # except exception as e: # removed try/except block
        # logging.warning(f"error reading language file for '{os.path.basename(song_dir_path)}': {e}")

    return languages

# function detect_languages removed as it's no longer used

# --- lyrics text extraction ---

def extract_lyrics_text(songs_dir_path):
    """extract lyrics text and language info from song directories."""
    # logging.info(f"extracting lyrics text from: {songs_dir_path}") # removed log

    lyrics_files = glob.glob(os.path.join(songs_dir_path, '*', 'lyrics.txt'))
    song_languages = {}  # dictionary for detected languages
    song_lyrics_map = {} # dictionary for raw lyrics text

    if not lyrics_files:
        # logging.warning(f"no lyrics.txt files found in subdirectories of {songs_dir_path}") # removed log
        return {}, {}  # return tuple of empty dictionaries

    processed_song_names = set() # track processed songs to avoid duplicates if structure is odd

    lyrics_count = 0
    empty_lyrics_count = 0

    for file_path in lyrics_files:
        # try: # removed outer try/except for path processing
        # extract song name from directory path
        song_name = os.path.basename(os.path.dirname(file_path))
        if song_name in processed_song_names:
            # logging.warning(f"duplicate song directory found or processed? skipping {file_path}") # removed log
            continue
        processed_song_names.add(song_name)

        lyrics = "" # default to empty string
        languages = [] # default to empty list

        # try reading lyrics first - standardized preprocessing
        # try: # removed inner try/except for lyrics reading
        with open(file_path, 'r', encoding='utf-8') as f:
            lyrics_content = f.read()

            # standardized preprocessing steps
            processed_lyrics = lyrics_content.strip()

            # basic check for empty or placeholder lyrics
            if processed_lyrics and "lyrics not found" not in processed_lyrics.lower():
                # apply consistent preprocessing
                processed_lyrics = processed_lyrics.replace('\r\n', '\n')  # normalize line endings

                lyrics = processed_lyrics
                lyrics_count += 1
            else:
                # logging.debug(f"skipping empty or placeholder lyrics for song: {song_name}") # removed log
                lyrics = "" # ensure it's an empty string if invalid
                empty_lyrics_count += 1
        # except exception as e: # removed inner try/except for lyrics reading
            # logging.warning(f"error reading lyrics file {file_path}: {e}. treating as empty lyrics.") # removed log
            # lyrics = ""
            # empty_lyrics_count += 1

        # always try reading language file, regardless of lyrics success
        # try: # removed inner try/except for language reading
        song_dir = os.path.dirname(file_path)
        languages = read_language_files(song_dir)
        # logging.debug(f"found languages for {song_name}: {languages}") # removed log
        # except exception as e: # removed inner try/except for language reading
             # logging.warning(f"error reading language file for {song_name} after lyrics processing: {e}") # removed log
             # languages = [] # ensure empty list on error

        # store results
        song_lyrics_map[song_name] = lyrics
        song_languages[song_name] = languages

        # except exception as e: # removed outer try/except for path processing
            # catch errors in path processing itself
             # logging.error(f"error processing path {file_path}: {e}") # removed log
             # attempt to extract song name for empty results if possible
             # try: # removed nested try/except
                 # song_name = os.path.basename(os.path.dirname(file_path))
                 # if song_name not in song_lyrics_map: song_lyrics_map[song_name] = ""
                 # if song_name not in song_languages: song_languages[song_name] = []
             # except exception: # removed nested try/except
                 # logging.error(f"could not even extract song name from path {file_path} during error handling.") # removed log

    # logging.info(f"extracted lyrics for {lyrics_count} songs, {empty_lyrics_count} had empty or invalid lyrics") # removed log
    return song_lyrics_map, song_languages


def find_most_similar(target_index, all_features, all_filepaths, song_word_sets_map, languages_dict, weights=None):
    """
    find similar songs using audio features, lyrics (word set jaccard), and language.

    combines audio (55%), lyrics (25% * 2 = 50%), and language (20%) similarity scores.
    note: lyrics weight is effectively doubled in the formula.
    """
    num_songs = len(all_features)
    if num_songs <= 1:
        # logging.info("not enough songs to find similarities (only 1 or 0 songs provided).") # removed log
        return []

    # --- 1. extract song names ---
    # use the helper function that extracts the directory name
    song_names = [_get_song_name_from_path(fp) for fp in all_filepaths]
    # check if extraction failed for any path
    if None in song_names:
         # logging.error("failed to extract song name from one or more filepaths. check path format. similarity results may be incomplete.") # removed log
         pass # attempt to proceed, but results might be affected

    target_song_name = song_names[target_index]
    if target_song_name is None:
        # logging.error(f"could not determine target song name for index {target_index} from path {all_filepaths[target_index]}. aborting similarity calculation for this target.") # removed log
        return [] # cannot proceed without target song name

    # --- 2. calculate audio similarity ---
    features_array = np.array(all_features)
    if weights is not None:
        # ensure weights is a numpy array for broadcasting
        weights = np.array(weights)
        # if weights.ndim != 1 or weights.shape[0] != features_array.shape[1]: # removed validation, assume correct if provided
            # raise valueerror(f"librosa weights shape {weights.shape} must be 1d and match feature dimension ({features_array.shape[1]})")
        weighted_features = features_array * weights
    else:
        weighted_features = features_array

    librosa_similarity_matrix = cosine_similarity(weighted_features)
    target_librosa_similarities = librosa_similarity_matrix[target_index] # similarities of target to all others

    # --- 4. calculate language similarity ---
    language_similarities = np.zeros(num_songs)

    # get target languages (needed for comparison)
    target_languages = languages_dict.get(target_song_name, [])
    # if not target_languages: # removed debug log
        # logging.debug(f"target song '{target_song_name}' has no languages listed in languages_dict.")

    for j in range(num_songs):
        comp_song_name = song_names[j]
        if comp_song_name is None:
            # logging.warning(f"skipping language similarity for index {j} due to missing song name.") # removed log
            language_similarities[j] = 0.0
            continue

        comp_languages = languages_dict.get(comp_song_name, [])

        # calculate language similarity (1.0 for shared language, 0.0 otherwise)
        if not target_languages or not comp_languages:
            # if either song lacks detected languages, default to 0.0
            language_similarities[j] = 0.0
        else:
            # check for at least one shared language
            shared_languages = set(target_languages).intersection(set(comp_languages))
            language_similarities[j] = 1.0 if shared_languages else 0.0

            # if shared_languages: # removed debug log
                # logging.debug(f"songs '{target_song_name}' and '{comp_song_name}' share languages: {shared_languages}")

    # --- 5. calculate final weighted similarity ---
    final_similarities = np.zeros(num_songs)

    # get target word set once
    target_word_set = song_word_sets_map.get(target_song_name, set())
    # if not target_word_set: # removed debug log
        # logging.debug(f"target song '{target_song_name}' has an empty word set.")

    for j in range(num_songs):
        comp_song_name = song_names[j]
        if j != target_index and comp_song_name is not None:
            audio_sim = target_librosa_similarities[j]
            lang_sim = language_similarities[j]

            # --- 3. calculate lyrics similarity (jaccard index on word sets) ---
            lyrics_sim = 0.0
            lyrics_method = "jaccard" # indicate the method used

            comp_word_set = song_word_sets_map.get(comp_song_name, set())

            if target_word_set and comp_word_set: # only calculate if both sets are non-empty
                intersection_size = len(target_word_set.intersection(comp_word_set))
                union_size = len(target_word_set.union(comp_word_set))
                if union_size > 0:
                    lyrics_sim = intersection_size / union_size
                # else: lyrics_sim remains 0.0 if union is 0 (both sets empty, already handled)
            # else: lyrics_sim remains 0.0 if one or both sets are empty

            # calculate final similarity
            final_sim = (0.55 * audio_sim) + 2 * (0.25 * lyrics_sim) + (0.2 * lang_sim)
            final_similarities[j] = final_sim

            # logging.info(f"similarity between '{target_song_name}' and '{comp_song_name}': audio: {audio_sim:.4f}, lyrics ({lyrics_method}): {lyrics_sim:.4f}, language: {lang_sim:.4f}, final: {final_sim:.4f}") # removed log

            # keep the print statement for backward compatibility
            print(f"  - weighted result: (0.4 * audio:{audio_sim:.4f}) + (0.4 * lyrics ({lyrics_method}):{lyrics_sim:.4f}) + (0.2 * lang:{lang_sim:.1f}) = {final_sim:.4f}")

    # --- 6. find top n recommendations ---
    # exclude the target song by setting its score low
    final_similarities[target_index] = -np.inf

    num_recommendations = min(5, num_songs - 1) # find top 5 or fewer
    if num_recommendations <= 0:
        return [] # should have been caught earlier, but double-check

    # get indices of the top n scores
    # np.argsort sorts ascending; take the last 'num_recommendations' indices
    top_indices = np.argsort(final_similarities)[-num_recommendations:]

    # --- 7. format results ---
    # create list of tuples: (song_name, final_similarity_score)
    # use the song_names list extracted earlier
    similar_items = []
    for i in top_indices:
        song_name = song_names[i]
        if song_name is not None: # ensure we have a valid song name
             similar_items.append((song_name, final_similarities[i]))
        # else: # removed logging for skipped result
            # logging.warning(f"skipping result for index {i} because song name could not be extracted from path: {all_filepaths[i]}")

    # sort results by final similarity score descending
    return sorted(similar_items, key=lambda item: item[1], reverse=True)


def calculate_all_similarities(all_features, all_filepaths, weights=None):
    """calculate similarity scores between all songs in the dataset."""
    num_songs = len(all_features)
    if num_songs <= 1:
        # logging.warning("not enough songs to calculate all similarities.") # removed log
        return {}

    all_similarities_map = {}
    song_names = [_get_song_name_from_path(fp) for fp in all_filepaths] # get all names once

    # --- pre-computation: extract lyrics, languages, and process with lyricsanalyzer ---
    lyrics_map = {}
    languages_dict = {} # dictionary for language info
    song_word_sets_map = {} # dictionary for word sets from the new analyzer

    # 1. derive base 'songs' directory
    derived_songs_base_dir = None
    if all_filepaths:
        # try: # removed try/except block
        first_path = all_filepaths[0]
        parts = first_path.split(os.sep)
        # handle potential valueerror if 'songs' is not in path
        if 'songs' in parts:
            songs_index = parts.index('songs')
            derived_songs_base_dir = os.sep.join(parts[:songs_index+1])
        # else: # removed error logging
            # logging.error(f"could not derive 'songs' base directory from path: {first_path}. cannot extract lyrics.")
        # except (valueerror, indexerror): # removed try/except block
             # logging.error(f"could not derive 'songs' base directory from path: {first_path}. cannot extract lyrics.") # removed log
    # else: # removed error logging
        # logging.error("no filepaths provided, cannot determine songs base directory for lyrics.")

    # 2. extract raw lyrics text and languages
    if derived_songs_base_dir:
        # logging.info(f"extracting lyrics text and languages from: {derived_songs_base_dir}") # removed log
        lyrics_map, languages_dict = extract_lyrics_text(derived_songs_base_dir)
        # if not lyrics_map: # removed warning log
            # logging.warning("no lyrics text was extracted. lyrics similarity will be zero.")
    # else: # removed warning log
         # logging.warning("could not determine base directory. lyrics similarity will be zero.")

    # 3. process lyrics with the new lyricsanalyzer to get word sets
    if lyrics_map:
        # logging.info("processing lyrics with lyricsanalyzer to generate word sets...") # removed log
        # try: # removed try/except block
        # initialize the lyricsanalyzer
        lyrics_analyzer = LyricsAnalyzer()
        # process all lyrics to get word sets
        for song_name, lyrics_text in lyrics_map.items():
            if lyrics_text and lyrics_text.strip():
                # construct the full path to the song directory
                song_dir_path = os.path.join(derived_songs_base_dir, song_name)
                song_word_sets_map[song_name] = lyrics_analyzer.analyze_lyrics(lyrics_text, song_dir_path)
            else:
                song_word_sets_map[song_name] = set() # store empty set for empty lyrics
        # logging.info(f"generated word sets for {len(song_word_sets_map)} songs.") # removed log
        # except exception as e: # removed try/except block
            # logging.error(f"error during lyrics analysis for word sets: {e}") # removed log
            # song_word_sets_map = {} # reset on error
    # else: # removed warning log
        # logging.warning("no lyrics map available, skipping lyrics analysis.")

    # --- calculate similarities for each song ---
    for i in range(num_songs):
        target_song_name = song_names[i]
        if target_song_name is None:
            # logging.warning(f"skipping similarity calculation for index {i} due to missing song name.") # removed log
            continue

        # logging.info(f"calculating similarities for: {target_song_name}") # removed log
        # pass the pre-calculated word sets map and languages dict
        similar_songs_data = find_most_similar(
            target_index=i,
            all_features=all_features,
            all_filepaths=all_filepaths,
            song_word_sets_map=song_word_sets_map, # pass the new word sets map
            languages_dict=languages_dict,
            weights=weights
        )

        # store the full (name, score) tuples
        all_similarities_map[target_song_name] = similar_songs_data

    return all_similarities_map


def write_similarities_to_csv(all_similarities, output_dir):
    """write song similarity results to csv files."""
    if not all_similarities:
        # logging.warning("no similarities data provided to write to csv.") # removed log
        return

    output_path = os.path.join(output_dir, 'song_similarities.csv')

    # sort dictionary items by song name (key) alphabetically
    sorted_items = sorted(all_similarities.items())

    # try: # removed try/except block
    # use 'w' mode to overwrite the file each time
    with open(output_path, 'w', encoding='utf-8') as f:
        # write the standard format file
        for main_song_name, recommendations in sorted_items:
            f.write(f"song: {main_song_name}\n") # write main song name
            if recommendations:
                for i, (rec_song_name, score) in enumerate(recommendations):
                    # format score as percentage string (e.g., 87.65%)
                    percentage_score = f"{score * 100:.2f}"
                    f.write(f"recommended song {i+1}: {rec_song_name} - {percentage_score}% similarity\n")
            else:
                f.write("no recommendations found.\n") # handle case with no recommendations

            f.write("\n") # write blank line after each song's recommendations

    # logging.info(f"successfully wrote formatted similarities to {output_path}") # removed log
    # except ioerror as e: # removed try/except block
        # logging.error(f"error writing similarities to csv file {output_path}: {e}") # removed log
    # except exception as e: # removed try/except block
        # logging.error(f"an unexpected error occurred while writing the csv: {e}") # removed log

if __name__ == '__main__':
    print("--- similarity calculation (combined audio + lyrics word sets) ---") # updated print message
    print("run main.py to execute the full similarity calculation pipeline.")