# Music Similarity Recommendation Project

## Project Overview

This project aims to recommend similar songs based on a multi-faceted analysis of their audio features, lyrical content, and language. Given a collection of songs, the system calculates pairwise similarity scores and identifies the top matches for each song. In essence, one would think that this aims to copy Spotify, and, well, it sort of does. It's not very robust as of now, but that was due to a lot of errors in calculations. Recommending a song is very hard, and as the author, I tweaked the ratios of the weights to see what the sweetest spot was for finding two similar songs by the same artist to be recommended. This was particularly difficult due to the lyrical analysis, in which artists' songs can have very little overlap in lyrical unison. 

### 1. Data Preparation (Lyrics Fetching)

*   Lyrics for each song are fetched using the Genius API via the `fetch_lyrics.py` script. Note: some of these songs were not unique names, and the songs I used had no artist attached, and thus sometimes the wrong lyrics were fetched. If this is the case, you should also correct it, or better yet, edit your song name to match the format "['Artist'] - ['Song_name']"
*   Fetched lyrics are stored in text files within each song's directory: `songs/SONG_NAME/lyrics.txt`.
*   Note: Initial audio processing steps like MP3-to-WAV conversion are considered deprecated in the current workflow, which assumes WAV files are already present. There were too many files in this workspace, so I removed it. There are converters online, however. 

### 2. Audio Feature Extraction - Librosa vs Essentia

*   The `feature_extractor.py` script utilizes the Essentia library to extract a comprehensive set of 44 low-level and high-level audio features from each `audio.wav` file.
*   Key features include Mel-Frequency Cepstral Coefficients (MFCCs), Harmonic Pitch Class Profiles (HPCP), loudness, and Beats Per Minute (BPM). 
*   Extracted features are scaled using `StandardScaler` from scikit-learn to ensure fair comparison.
*   The scaled features for all songs are aggregated and stored in `audio_features.pkl`.

**Deprecated Method:**
*   The Librosa library was considered, and it was in fact a part of the preliminary project proposal. However, this had been left unchanged for a very long time, as I was unsure if there was a backup to some of the features it contained. The central problem with Librosa was that it was returning consistently high values across the board of every song comparison, that is, almost all 'audio' components were returning {0.89, 0.93, 0.98}, and contained very little variance in between (and on the lower end). As a result of some searching, we ended up going with the Essentia library (with it's stereo vs mono pickiness) and the spectrum of [-1, 1] seems to be much better.

### 3. Lyrical Feature Analysis

Two approaches to lyrical analysis have been considered:

*   **Current Method (Implemented):**
    *   Managed by `lyrics_analyzer.py`.
    *   Lyrics undergo cleaning: conversion to lowercase, removal of punctuation, and filtering of common stopwords (taken from online).
    *   A set of unique, meaningful words is created for each song's lyrics.
    *   These unique word sets are cached in `lyrics_analysis.pkl` using Pickle.
    *   The `similarity_calculator.py` script compares these word sets using Jaccard similarity to determine lyrical similarity based on shared vocabulary.

*   **Advanced Method - BERT (Previous Iteration):**
    *   *This method is **not** the currently active implementation in the analyzed code but represents a past exploration.*
    *   Utilized DistilBERT embeddings to capture the semantic meaning of lyrics, going beyond simple word matching.
    *   Employed zero-shot emotion classification to identify emotional tones across 13 categories (e.g., {joy, sadness, anger, fear, love})
    *   Used ConceptNet to expand keywords with related concepts, specially for sparser lyrics. Example, "ocean" would branch out to "water", "sea", etc.
    * This (unfortunately) was a previous iteration because it was too costly to retrain over and over, and I was unsure on how to handle cache optimization, as it goes a bit beyond the scope of my abilities to handle possibly training a transformer into the mix. The choice to remove it was for the implementation, as I understand the power that was (or could have been) leveragable, possibly returning better values if it had been trained well. The re-introduction of this is "left as an exercise to the reader"!

### 4. Language Feature Detection

*   The `generate_language_files.py` script uses the `langdetect` library to identify the language(s) present in each `lyrics.txt` file. This wasn't perfect, and so some manual intervention was done to verify all of them. About 80% of them were correct, especially since some songs had no lyrics and thus were inferred as a random language. 
*   The detected language code(s) are stored in `language.txt` within each song's directory.
*   In `similarity_calculator.py`, this information is used to calculate a binary language match score: 1 if two songs share at least one detected language, 0 otherwise. This was useful since some songs had multiple songs in them (a bilingual song), and so that they could span across more than just one of their languages. It was also useful for preventing having to check for a "main" language, but off the top of my head that could have been done based off of ratios of the language(s) present: majority wins.

### 5. Combined Similarity Score

*   The final similarity score between any two songs is calculated in `similarity_calculator.py`.
*   It's a weighted sum of the individual similarity components:
    *   `Similarity = (0.55 * Audio Similarity) + (0.50 * Lyrics Similarity) + (0.20 * Language Similarity)`
    *   Audio Similarity is based on the cosine distance between scaled Essentia feature vectors- there might be a warning about type conversions, I tried to address them, it didn't seem to make the warning go away, but it still calculated everything out after some tuple extraction. This is based off of a range from [-1,1]
    *   Lyrics Similarity is based on the Jaccard similarity of unique word sets-- the shared words among both songs not including the stop words from online. This is based off of a range from [0,1]
    *   Language Similarity is the binary (0 or 1) of whether two songs share any of the same languages in their lists (if either include more than 1).

## How to Run

1.  **Clone the Repository:**
    ```bash
    git clone <https://github.com/brosca2/521-music-rec>
    cd 521-music-rec
    ```
2.  **Environment Variables:** Create a `.env` file in the `521-music-rec` directory with your Genius API key:
    ```
    GENIUS_API_KEY=YOUR_KEY_HERE
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Prepare Songs:** Ensure your songs are organized in the following structure, with audio files in WAV format: (you generally can find a tool online to paste an entire Spotify playlist to get a folder of mp3s, then route that into an mp3-to-wav converter. I've uploaded my songs, of which were the result of my playlist):
    ```
    songs/
    ├── SONG_NAME_1/
    │   └── audio.wav
    ├── SONG_NAME_2/
    │   └── audio.wav
    └── ...
    ```

### Execution

1.  **Fetch Lyrics:** Run the lyrics fetching script (requires the `.env` file). Edit whatever songs as needed.
    ```bash
    python fetch_lyrics.py --songs_dir songs
    ```
2.  **Generate Language Files:** Run the language detection script. Edit whatever languages as needed.
    ```bash
    python generate_language_files.py --songs_dir songs
    ```
3.  **Run Main Pipeline:** Execute the main script to perform feature extraction and similarity calculation. It will print results to the console and also to a .csv
    ```bash
    python main.py --songs_dir songs
    ```

## Output

The script calculates pairwise similarities for all the songs in the `songs` directory. The final recommendations are saved to `song_similarities.csv`. This file lists each source song and its top 5 most similar songs from the collection, ranked by the combined similarity score. Multiple trial and error attempts have proven that results can be *extremely* different when tweaking the weights. 
