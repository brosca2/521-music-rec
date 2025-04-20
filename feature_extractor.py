import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import essentia.standard as es


def extract_essentia_features(file_path):
    """
    extracts specific audio features using essentia, utilizing pickle caching.

    checks for a cached '.pkl' file in the song's directory first. if found and valid,
    loads features from cache. otherwise, extracts features using essentia and saves
    them to the cache file.

    args:
        file_path (str): path to the .wav audio file.

    returns:
        numpy.ndarray or none: a numpy array containing extracted features
                               [integratedloudness, loudnessrange, danceability, bpm, dynamiccomplexity].
                               returns none if processing fails or features are invalid.
    """
    # caching logic
    audio_dir = os.path.dirname(file_path)
    cache_file_name = 'essentia_features.pkl'
    cache_file_path = os.path.join(audio_dir, cache_file_name)

    # check for cache
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'rb') as f:
            cached_features = pickle.load(f)
            # basic validation: check numpy array shape (44 features)
            # basic validation: check numpy array shape (44 features)
            if isinstance(cached_features, np.ndarray) and cached_features.shape == (44,):
                return cached_features
            # else: # invalid cache data, proceed to extraction


    # original feature extraction logic (if cache miss or load error)
    features = {}
    # load audio using audioloaders (handles various formats)
    loader = es.AudioLoader(filename=file_path)
    audio_data, sample_rate, num_channels, _, _, _ = loader()
    
    # keep original stereo format for algorithms expecting stereo input
    # most essentia algorithms for feature extraction expect stereo input
    audio = audio_data
    
    # proceed with feature extraction using original audio format

    # convert stereo to mono for algorithms requiring it
    if audio.shape[1] > 1:  # if stereo
        # ensure float32 type
        mono_audio = np.mean(audio, axis=1).astype(np.float32)
    else:  # if already mono
        # ensure 1d and float32
        mono_audio = audio[:, 0].astype(np.float32)

    # frame-based processing setup
    frameSize = 2048
    hopSize = 1024
    sampleRate = sample_rate # use sample rate from loader

    # initialize essentia algorithms for frame processing
    frame_generator = es.FrameGenerator(audio=mono_audio, frameSize=frameSize, hopSize=hopSize, startFromZero=True)
    window = es.Windowing(type='hann', zeroPadding=0) # no zero padding typically needed for these features
    spectrum = es.Spectrum() # standard spectrum only outputs magnitude
    spectral_centroid_time = es.SpectralCentroidTime(sampleRate=sampleRate)
    hpcp_algo = es.HPCP(size=12, sampleRate=sampleRate) # 12 bins for hpcp
    mfcc_algo = es.MFCC(numberCoefficients=13, sampleRate=sampleRate) # first 13 mfccs

    # calculate frequencies for fft bins (constant for all frames)
    # number of bins in rfft output is framesize // 2 + 1
    frequencies = np.fft.rfftfreq(n=frameSize, d=1.0/sampleRate)
    # lists to store frame-wise results
    centroid_frames = []
    hpcp_frames = []
    mfcc_frames = [] # store coefficient vectors per frame

    # process audio frame by frame
    for frame in frame_generator:
        frame_windowed = window(frame)
        magnitudes = spectrum(frame_windowed) # spectrum now only returns magnitudes
        magnitudes = magnitudes.astype(np.float32) # convert to float32 for essentia compatibility

        # spectral centroid
        centroid_output = spectral_centroid_time(frame_windowed)
        centroid = centroid_output # centroid works on time-domain windowed frame
        centroid_frames.append(centroid)

        # hpcp
        # hpcp requires spectrum magnitude and pre-calculated frequency vector
        hpcp_output = hpcp_algo(magnitudes, frequencies) # use pre-calculated frequencies
        hpcp_values = hpcp_output
        hpcp_frames.append(hpcp_values) # append the 12 hpcp values for this frame

        # mfcc
        # mfcc requires spectrum magnitude
        mfcc_output = mfcc_algo(magnitudes) # use magnitudes from spectrum
        _, mfcc_coeffs = mfcc_output # use the calculated magnitudes
        mfcc_frames.append(mfcc_coeffs) # append the 13 coefficients for this frame

    # aggregate frame-wise features
    features['spectral_centroid_mean'] = np.mean(centroid_frames) if centroid_frames else 0.0

    if hpcp_frames:
        hpcp_matrix = np.array(hpcp_frames)
        hpcp_means = np.mean(hpcp_matrix, axis=0)
    else:
        hpcp_means = np.zeros(12) # default if no frames
    for i in range(12):
        features[f'hpcp_mean_{i}'] = hpcp_means[i]

    if mfcc_frames:
        mfcc_matrix = np.array(mfcc_frames)
        mfcc_means = np.mean(mfcc_matrix, axis=0)
        mfcc_stds = np.std(mfcc_matrix, axis=0)
    else:
        mfcc_means = np.zeros(13) # default if no frames
        mfcc_stds = np.zeros(13)  # default if no frames
    for i in range(13):
        features[f'mfcc_mean_{i}'] = mfcc_means[i]
        features[f'mfcc_std_{i}'] = mfcc_stds[i]

    # original feature extraction (loudness, danceability, bpm, dynamic complexity)

    # 1. loudness (ebu r128) - requires stereo input
    if num_channels == 2:
        loudness_ebu = es.LoudnessEBUR128()
        # use the original stereo audio data
        loudness_results = loudness_ebu(audio)
        # access correct indices for scalar loudness values
        features['integratedLoudness'] = loudness_results[2] # integrated loudness
        features['loudnessRange'] = loudness_results[3]    # loudness range
        # handle potential loudness calculation failure by assigning defaults if needed
        if features['integratedLoudness'] is None: features['integratedLoudness'] = 0.0 # default on error
        if features['loudnessRange'] is None: features['loudnessRange'] = 0.0 # default on error
    else:
        # assign default values if audio is not stereo
        features['integratedLoudness'] = 0.0
        features['loudnessRange'] = 0.0

    # 2. danceability - expects mono input
    danceability_algo = es.Danceability()
    danceability_results = danceability_algo(mono_audio)
    features['danceability'] = danceability_results[0]

    # 3. rhythm (bpm) - expects mono input
    rhythm_algo = es.RhythmExtractor2013()
    rhythm_results = rhythm_algo(mono_audio)
    # check if bpm result is scalar and positive
    if isinstance(rhythm_results[0], (int, float)) and rhythm_results[0] > 0:
        bpm = rhythm_results[0]
    else:
        bpm = 0.0 # default to 0.0 if invalid or not a number
    features['bpm'] = bpm
    # handle potential rhythm extraction failure by assigning default if needed
    if features['bpm'] is None: features['bpm'] = 0.0 # assign default on error


    # 4. dynamic complexity - expects mono input
    dynamic_complexity_algo = es.DynamicComplexity()
    dynamic_complexity_results = dynamic_complexity_algo(mono_audio)
    features['dynamicComplexity'] = dynamic_complexity_results[0]

    # validation: ensure all features are scalar numbers
    # define all expected feature keys
    feature_keys = [
        'integratedLoudness', 'loudnessRange', 'danceability', 'bpm', 'dynamicComplexity',
        'spectral_centroid_mean'
    ]
    feature_keys.extend([f'hpcp_mean_{i}' for i in range(12)])
    feature_keys.extend([f'mfcc_mean_{i}' for i in range(13)])
    feature_keys.extend([f'mfcc_std_{i}' for i in range(13)])

    for key in feature_keys:
        value = features.get(key) # use .get() for safety
        # allow none temporarily if feature failed, check later
        if value is None:
             features[key] = 0.0 # replace none with 0.0 to avoid type errors later
             value = 0.0
        elif not isinstance(value, (int, float, np.number)): # check if scalar number
            # invalid feature type, return none to indicate failure
            return None

    # create numpy array only if all features are valid scalars
    # build feature vector in defined order
    feature_values = [
        features['integratedLoudness'],
        features['loudnessRange'],
        features['danceability'],
        features['bpm'],
        features['dynamicComplexity'],
        features['spectral_centroid_mean']
    ]
    feature_values.extend([features[f'hpcp_mean_{i}'] for i in range(12)])
    feature_values.extend([features[f'mfcc_mean_{i}'] for i in range(13)])
    feature_values.extend([features[f'mfcc_std_{i}'] for i in range(13)])

    feature_vector = np.array(feature_values, dtype=np.float32) # ensure float type

    # check for nan or inf values which can break standardscaler
    if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
        # invalid values found, return none
        return None

    # final check for expected vector length (44 features)
    expected_length = 44
    if feature_vector.shape[0] != expected_length:
        # unexpected length, return none
        return None

    # caching logic: save features before returning
    with open(cache_file_path, 'wb') as f:
        pickle.dump(feature_vector, f)

    return feature_vector


def process_audio_files(songs_dir='songs', output_file='audio_features.pkl'):
    """
    extracts features for all audio files in the specified directory,
    applies standardscaler, and saves the results.

    args:
        songs_dir (str): directory containing subdirectories for each song.
        output_file (str): path to save the final features dictionary (pickle file).
    """
    all_features_raw = []
    song_paths_map = {} # map index back to song name/path after scaling
    valid_indices = [] # track indices of successfully processed songs

    song_index = 0
    for song_name in sorted(os.listdir(songs_dir)):
        song_dir_path = os.path.join(songs_dir, song_name)
        if os.path.isdir(song_dir_path):
            audio_file_path = os.path.join(song_dir_path, 'audio.wav')
            if os.path.exists(audio_file_path):
                features = extract_essentia_features(audio_file_path)
                if features is not None:
                    all_features_raw.append(features)
                    # use song_name as key (assumed unique)
                    song_paths_map[song_index] = song_name
                    valid_indices.append(song_index)
                    song_index += 1
                # else: # skip song due to error

    if not all_features_raw:
        return

    # convert list of feature vectors to numpy array for scaling
    features_matrix_raw = np.array(all_features_raw)

    # apply standardscaler
    scaler = StandardScaler()
    features_matrix_scaled = scaler.fit_transform(features_matrix_raw)

    # create final dictionary mapping song names to scaled features
    final_features = {}
    for i, scaled_vector in enumerate(features_matrix_scaled):
        original_index = valid_indices[i] # get original index before potential skips
        song_name = song_paths_map[original_index]
        final_features[song_name] = scaled_vector

    # save the scaled features
    with open(output_file, 'wb') as f:
        pickle.dump(final_features, f)


# example usage (optional, can be called from main.py)
if __name__ == "__main__":
    # allows running feature extraction independently if needed
    # but main.py should ideally orchestrate this.
    # warnings.warn("running feature_extractor.py directly. this is usually handled by main.py.") # removed warning
    process_audio_files(songs_dir='songs', output_file='audio_features.pkl')

# end of feature extraction module