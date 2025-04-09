import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_most_similar(target_index, all_features, all_filepaths, weights=None):
    """
    finds the top N (currently 5) most similar songs to a target song based on feature vectors.

    args:
        target_index (int): the index of the target song in all_features and all_filepaths.
        all_features (list or np.ndarray): a list or array where each element is a feature vector (np.ndarray) for a song.
        all_filepaths (list): a list of file paths corresponding to the features, in the same order.
        weights (np.ndarray, optional): an array of weights to apply to each feature dimension during similarity calculation.
                                        defaults to none (equal weighting).

    returns:
        list: a list of tuples, sorted by similarity score in descending order.
              each tuple contains (trimmed_filepath, similarity_score).
              returns the top 5 most similar songs (excluding the target itself).
              returns fewer if less than 6 songs (target + 5 others) are available.
              returns an empty list if only the target song exists.
    """
    num_songs = len(all_features)

    # handle edge case: not enough songs to find 5 recommendations
    # if num_songs <= 1 (only target exists), the loop below won't find anything, returning [] correctly.
    # if 1 < num_songs <= 5, it will return all *other* songs.
    # the logic below handles finding the top N up to num_songs - 1.

    # ensure all_features is a numpy array for efficient processing with sklearn

    features_array = np.array(all_features)

    # apply feature weights if provided
    if weights is not None:
        # ensure weights is a numpy array
        weights = np.array(weights)
        # validate that the weights array has the same number of elements as features per song
        if weights.ndim != 1 or weights.shape[0] != features_array.shape[1]:
             raise ValueError(f"weights shape {weights.shape} must be a 1d array matching the feature dimension ({features_array.shape[1]})")
        # apply weights using element-wise multiplication (broadcasting)
        weighted_features = features_array * weights
    else:
        # use original features if no weights are specified
        weighted_features = features_array

    # calculate the pairwise cosine similarity between all songs using the weighted (or original) features
    # result is a matrix where similarity_matrix[i, j] is the similarity between song i and song j
    similarity_matrix = cosine_similarity(weighted_features)

    # extract the row corresponding to the target song's similarities with all other songs
    target_similarities = similarity_matrix[target_index]

    # exclude the target song itself from the recommendations
    # set its similarity score to negative infinity so it's never chosen as the most similar
    target_similarities[target_index] = -np.inf
    
    # find the indices of the songs with the highest similarity scores
    # np.argsort sorts in ascending order, so we take the last 5 indices for the top 5 scores
    # if num_songs is small, this will correctly take fewer than 5 indices
    num_recommendations = min(5, num_songs - 1) # find top 5 or fewer if not enough songs
    if num_recommendations <= 0:
        return [] # return empty list if only the target song exists

    top_indices = np.argsort(target_similarities)[-num_recommendations:]

    # helper function to trim the specific prefix from filenames for cleaner display
    def trim_filename(filepath):
        prefix = '[SPOTDOWNLOADER.COM] '
        # get the actual filename part, handling both unix and windows path separators
        basename = filepath.split('/')[-1].split('\\')[-1]
        if basename.startswith(prefix):
            # return the filename *without* the prefix
            # note: this returns only the basename, not the modified full path
            return basename[len(prefix):]
        # return the original basename if the prefix is not found
        return basename

    # create a list of tuples: (trimmed_filename, similarity_score) for the top songs
    similar_items = [(trim_filename(all_filepaths[i]), target_similarities[i]) for i in top_indices]

    # sort the results by similarity score in descending order
    return sorted(similar_items, key=lambda item: item[1], reverse=True)

# example usage block (runs only when the script is executed directly)
if __name__ == '__main__':
    # --- dummy data for testing ---
    dummy_features = [
        np.array([0.1, 0.9, 0.2]),    # Song 0
        np.array([0.8, 0.1, 0.3]),    # Song 1
        np.array([0.15, 0.85, 0.25]), # Song 2 (Similar to 0)
        np.array([0.7, 0.2, 0.4])     # Song 3 (Similar to 1)
    ]
    # example weights: emphasize the first feature more than the others
    dummy_weights = np.array([2.0, 1.0, 1.0])
    dummy_filepaths = [
        'songs/[SPOTDOWNLOADER.COM] song0.wav', # example with prefix
        'songs/song1.wav',                     # example without prefix
        'songs/[SPOTDOWNLOADER.COM] song2.wav',
        'songs/song3.wav'
    ]
    target_song_index = 0
    
    # --- test cases ---
    print("--- testing similarity calculation ---")

    # test without weights (filenames should be trimmed in the output)
    target_display_name_0 = trim_filename(dummy_filepaths[target_song_index]) # use the helper for display
    similar_songs_unweighted = find_most_similar(target_song_index, dummy_features, dummy_filepaths)
    print(f"\nsongs most similar to '{target_display_name_0}' (unweighted):")
    for name, score in similar_songs_unweighted:
        print(f"  - {name} (score: {score:.4f})")

    # test with weights
    similar_songs_weighted = find_most_similar(target_song_index, dummy_features, dummy_filepaths, weights=dummy_weights)
    print(f"\nsongs most similar to '{target_display_name_0}' (weighted {dummy_weights}):")
    for name, score in similar_songs_weighted:
        print(f"  - {name} (score: {score:.4f})")

    target_song_index = 1
    target_display_name_1 = trim_filename(dummy_filepaths[target_song_index])
    similar_songs = find_most_similar(target_song_index, dummy_features, dummy_filepaths) # unweighted test
    print(f"\nsongs most similar to '{target_display_name_1}' (unweighted):")
    for name, score in similar_songs:
        print(f"  - {name} (score: {score:.4f})")

    # test edge case (only 2 songs total)
    print("\n--- testing edge cases ---")
    dummy_features_short = [np.array([0.1, 0.9]), np.array([0.2, 0.8])]
    dummy_filepaths_short = ['short/[SPOTDOWNLOADER.COM] songA.wav', 'short/songB.wav']
    target_display_name_short = trim_filename(dummy_filepaths_short[0])
    # the function should return only the *other* song
    similar_songs_short_info = find_most_similar(0, dummy_features_short, dummy_filepaths_short)
    print(f"\nsongs most similar to '{target_display_name_short}' (2 total songs):")
    for name, score in similar_songs_short_info:
        print(f"  - {name} (score: {score:.4f})")

    
    # test edge case (only 1 song total)
    dummy_features_single = [np.array([0.1, 0.9])]
    dummy_filepaths_single = ['single/[SPOTDOWNLOADER.COM] songX.wav']
    target_display_name_single = trim_filename(dummy_filepaths_single[0])
    # the function should return an empty list
    similar_songs_single = find_most_similar(0, dummy_features_single, dummy_filepaths_single)
    print(f"\nsongs most similar to '{target_display_name_single}' (1 total song): {similar_songs_single}")