import os
import argparse
import multiprocessing
import numpy as np
# import logging # removed logging import
import pickle
from feature_extractor import process_audio_files # updated import
from similarity_calculator import calculate_all_similarities, write_similarities_to_csv
from similarity_calculator import extract_lyrics_text # removed unused imports
from lyrics_analyzer import LyricsAnalyzer
# removed minmaxscaler as scaling is now done in feature_extractor

# configure logging # removed logging configuration
# logging.basicconfig(level=logging.info, format='%(asctime)s - %(levelname)s - %(message)s')

# define the cache file path
CACHE_FILE = "audio_features.pkl" # use relative path within 521-music-rec dir

# removed process_song helper function as parallel processing is removed

def main():
    """
    main function to find similar songs based on audio features.
    parses args for songs directory, extracts features from .wav files,
    scales features, calculates similarity, and writes results to csv.
    """
    parser = argparse.ArgumentParser(description="find similar songs based on audio features.")
    parser.add_argument("--songs_dir", required=True, help="path to the base directory containing song subdirectories (e.g., 'songs').")
    # removed --limit argument as feature extraction handles all songs now

    args = parser.parse_args()

    # validate the songs directory path
    if not os.path.isdir(args.songs_dir):
        print(f"error: songs directory not found or is not a directory: {args.songs_dir}")
        return

    print(f"running feature extraction using essentia (will create/update {CACHE_FILE})...")
    # call the unified feature extraction function from feature_extractor
    # this function handles finding files, extracting, scaling, and saving the cache.
    process_audio_files(songs_dir=args.songs_dir, output_file=CACHE_FILE)
    print("feature extraction process complete.")

    # --- load features from the cache file created by process_audio_files ---
    print(f"loading processed features from {CACHE_FILE}...")
    if not os.path.exists(CACHE_FILE):
        print(f"error: cache file {CACHE_FILE} not found after feature extraction.")
        return

    # try: # removed try/except block
    with open(CACHE_FILE, 'rb') as f:
        # the cache now contains a dictionary: {'song_name': scaled_feature_vector}
        features_dict = pickle.load(f)
    if not features_dict:
        print("error: no features found in the cache file.")
        return
    print(f"successfully loaded features for {len(features_dict)} songs from {CACHE_FILE}.")
    # except exception as e: # removed try/except block
        # print(f"error loading features from cache file {cache_file}: {e}")
        # return

    # --- reconstruct ordered lists for similarity calculation ---
    # sort by song name to ensure consistent order
    sorted_song_names = sorted(features_dict.keys())

    all_filepaths = []
    all_features_list = []
    for song_name in sorted_song_names:
        # construct the expected full path to the audio file
        # assumes structure: args.songs_dir / song_name / audio.wav
        file_path = os.path.join(args.songs_dir, song_name, 'audio.wav')
        all_filepaths.append(file_path)
        all_features_list.append(features_dict[song_name])

    # convert list of feature vectors to a numpy array
    scaled_features = np.array(all_features_list)

    if not scaled_features.size > 0:
         print("warning: no features loaded or reconstructed. cannot proceed with similarity calculation.")
         return # exit if no features to compare

    print("\n--- processing lyrics and calculating all song similarities ---")

    # define feature weights (optional) - same logic
    feature_weights = None
    if feature_weights is not None and scaled_features.shape[1] > 0: # check scaled_features shape
        num_features_extracted = scaled_features.shape[1] # get dimension from numpy array
        if feature_weights.shape[0] != num_features_extracted:
            print(f"warning: provided weights shape {feature_weights.shape} does not match extracted feature dimension ({num_features_extracted}). ignoring weights.")
            feature_weights = None

    # extract lyrics from song directories
    print("extracting lyrics from song directories...")
    # use args.songs_dir directly for lyrics extraction, assuming it's the base 'songs' directory
    lyrics_map, languages_dict = extract_lyrics_text(args.songs_dir)
    print(f"found lyrics for {len(lyrics_map)} songs")

    # lyrics processing now handled inside calculate_all_similarities

    # calculate all similarities at once
    # calculate_all_similarities handles lyrics extraction/processing internally
    all_similarities = calculate_all_similarities(
        scaled_features,
        all_filepaths,
        weights=feature_weights
        # no longer need to pass song_lyrics_features
    )

    if not all_similarities:
        print("error: failed to calculate any similarities.")
        return

    print("\n--- writing similarities to csv ---")

    # determine output directory
    # output csv files to the current directory (where main.py is run from)
    csv_output_directory = "."
    print(f"writing csv output to the current directory ('{csv_output_directory}')")

    write_similarities_to_csv(all_similarities, csv_output_directory)

    print(f"\nprocessing complete. similarity results saved to '{os.path.join(csv_output_directory, 'song_similarities.csv')}'")


# main execution block: runs only when script executed directly
if __name__ == "__main__":
    main()