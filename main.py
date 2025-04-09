import os
import argparse
import multiprocessing
import numpy as np
from feature_extractor import extract_features
from similarity_calculator import find_most_similar # Already updated for weights and trimming
from sklearn.preprocessing import MinMaxScaler

# helper function for parallel processing: extracts features for one song
def process_song(file_path):
    """extracts features for a single song file."""
    filename = os.path.basename(file_path)
    try:
        features = extract_features(file_path)
        if features is not None:
            # print(f"successfully extracted features for {filename}.") # moved print outside
            return file_path, features
        else:
            # print(f"warning: could not extract features for {filename}. skipping.") # moved print outside
            return file_path, None
    except Exception as e:
        # print(f"error processing {filename}: {e}. skipping.") # moved print outside
        # return the error message string to distinguish from successful 'None' returns
        return file_path, f"Error: {e}"
# decided to keep this, shows full train of production, I downloaded my entire playlist from spotify using 
# its share link, and inputted that to SPOTDOWNLOADER.COM, which gave me the .webm
def trim_filename_main(filepath):
    """trims the specific prefix '[SPOTDOWNLOADER.COM] ' from a filename's basename."""
    prefix = '[SPOTDOWNLOADER.COM] '
    basename = os.path.basename(filepath)
    if basename.startswith(prefix):
        # return only the part of the filename *after* the prefix
        # note: this differs slightly from the calculator's trim which might modify the full path
        # for display in main, just modifying the basename is sufficient
        return basename[len(prefix):]
        # If full path modification is needed:
        # return filepath.replace(prefix, '', 1) # alternative: modify full path
    return basename # return original basename if prefix wasn't found

def main():
    """
    main function to find similar songs based on extracted features.
    parses command-line arguments for the songs directory, extracts features
    from all .wav files in parallel, scales features, calculates similarity
    for each song against all others, and prints recommendations.
    """
    parser = argparse.ArgumentParser(description="Find similar songs based on audio features.")
    parser.add_argument("--songs_dir", required=True, help="Path to the directory containing .wav song files.")
    # parser.add_argument("--target_song", required=True, help="filename of the target song within the songs directory.") # removed target_song argument

    args = parser.parse_args()

    # validate the provided songs directory path
    if not os.path.isdir(args.songs_dir):
        print(f"Error: Songs directory not found or is not a directory: {args.songs_dir}")
        return

    # target song validation removed as we now iterate through all songs

    print(f"Processing songs in directory: {args.songs_dir}")
    # print(f"target song: {args.target_song}") # removed target_song print

    all_features = []
    all_filepaths = []
    wav_files = [f for f in os.listdir(args.songs_dir) if f.lower().endswith('.wav')]

    if not wav_files:
        print(f"Error: No .wav files found in the directory: {args.songs_dir}")
        return

    # --- parallel feature extraction ---
    num_songs = len(wav_files)
    print(f"\nStarting feature extraction for {num_songs} songs using multiprocessing...")

    # use a context manager for the multiprocessing pool to ensure proper cleanup
    results_async = []
    # determine number of processes - use cpu_count() or a fixed number
    # example: use half the available cpu cores, but at least 1
    num_processes = max(1, multiprocessing.cpu_count() // 2)
    print(f"using {num_processes} worker processes.")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        for i, filename in enumerate(wav_files):
            file_path = os.path.join(args.songs_dir, filename)
            print(f"processing song #{i+1}/{num_songs}: {filename}") # simple progress indicator
            # asynchronously apply the process_song function to the file path
            async_result = pool.apply_async(process_song, (file_path,))
            results_async.append(async_result) # store the async result object

        # prevent new tasks from being submitted
        pool.close()
        # wait for all worker processes to finish
        pool.join()

    print("\nFeature extraction complete. Processing results...")
    processed_count = 0
    error_count = 0
    for result in results_async:
        try:
            file_path, features_or_error = result.get()
            if isinstance(features_or_error, np.ndarray):
                all_features.append(features_or_error)
                all_filepaths.append(file_path)
                processed_count += 1
                # optional: print success per file here if needed
                # print(f"successfully processed: {os.path.basename(file_path)}")
            elif isinstance(features_or_error, str) and features_or_error.startswith("Error:"):
                 print(f"Error processing {os.path.basename(file_path)}: {features_or_error}")
                 error_count += 1
            else: # should be none if feature extraction failed gracefully within process_song
                print(f"warning: could not extract features for {os.path.basename(file_path)}. skipping.")
                error_count += 1
        except Exception as e:
            # this catches errors during the result.get() call itself (less likely)
            print(f"error retrieving result for a task: {e}")
            error_count += 1

    print(f"\nProcessed {processed_count} songs successfully, {error_count} songs failed or were skipped.")

    if not all_filepaths:
        print("Error: No features could be extracted from any song.")
        return
    # convert the list of feature arrays into a single 2d numpy array
    all_features_np = np.array(all_features)

    # --- feature scaling ---
    # scale features to the range [0, 1] using minmaxscaler
    # this prevents features with larger ranges from dominating the distance calculation
    scaled_features = all_features_np # default to original if scaling fails or is skipped
    if all_features_np.size > 0:
        print("\nScaling features...")
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(all_features_np)
        print("Features scaled.")
    else:
        print("warning: skipping scaling as no features were extracted.")
        return # exit if no features to scale/compare

    print("\n--- finding similar songs for each successfully processed track ---")

    # define feature weights (optional)
    # set to none to use equal weighting (default in similarity_calculator)
    feature_weights = None
    # example: if features had 31 dimensions, you could define weights like:
    # feature_weights = np.array([1.0, 1.5, 1.0, ..., 2.0]) # must have 31 elements
    # ensure the shape matches the actual number of features extracted (31 in this case)
    if feature_weights is not None and len(all_features) > 0:
        num_features_extracted = all_features[0].shape[0]
        if feature_weights.shape[0] != num_features_extracted:
            print(f"warning: provided weights shape {feature_weights.shape} does not match extracted feature dimension ({num_features_extracted}). ignoring weights.")
            feature_weights = None # reset to none if shape mismatch


    # iterate through each successfully processed song and find its recommendations
    for target_index, target_song_path in enumerate(all_filepaths):
        # trim filename prefix for display purposes
        target_song_display_name = trim_filename_main(target_song_path)
        print(f"\nCalculating similarities for: '{target_song_display_name}'")

        # find recommendations for the current target song using the scaled features
        # pass the optional weights to the similarity function
        recommendations = find_most_similar(target_index, scaled_features, all_filepaths, weights=feature_weights)

        print(f"Recommendations for '{target_song_display_name}':")
        if not recommendations:
            print("  No similar songs found (excluding the target song itself).")
        else:
            # display top n recommendations (e.g., top 5)
            num_recommendations_to_show = min(len(recommendations), 5)
            for i in range(num_recommendations_to_show):
                rec_path, rec_score = recommendations[i]
                # the 'rec_path' returned by find_most_similar should already be the trimmed filename
                # (as handled by the trim_filename function within similarity_calculator.py)
                print(f"  {i+1}. {rec_path} (similarity: {rec_score:.4f})") # use rec_path directly
                # print(f"{i+1}. {rec_filename} (similarity: {rec_score:.4f})") # removed duplicate print

# main execution block: runs only when the script is executed directly
if __name__ == "__main__":
    main()