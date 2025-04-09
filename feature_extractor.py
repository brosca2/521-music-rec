import librosa
import numpy as np
import warnings

def extract_features(file_path):
    """
    extracts audio features from a .wav file using librosa.

    args:
        file_path (str): path to the .wav audio file.

    returns:
        numpy.ndarray or None: a flat numpy array containing the extracted features
                               (tempo, zcr, centroid, bandwidth, rms, beat_std,
                                13 mfccs, 12 chroma) if successful, otherwise none.
                                the array will have a shape of (31,).
    """
    try:
        # suppress specific librosa warnings (e.g., about audioread) during loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            # Load the audio file
            # load the audio time series (y) and its original sample rate (sr)
            y, sr = librosa.load(file_path, sr=None)

        # 1. tempo and beat tracking
        # tempo: estimated global tempo in beats per minute (bpm)
        # beat_frames: frame indices corresponding to detected beat events
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        # 2. beat regularity (standard deviation of beat intervals)
        # convert beat frame indices to time in seconds
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        # calculate the time difference between consecutive beats
        beat_intervals = np.diff(beat_times)
        if len(beat_intervals) > 1:
            beat_interval_std = np.std(beat_intervals)
        else:
            # handle cases with 0 or 1 beats detected (resulting in 0 or 1 interval)
            # standard deviation is not meaningful in these cases, so set to 0
            beat_interval_std = 0.0


        # 3. zero-crossing rate (rate at which the signal changes sign)
        # often correlates with the noisiness or percussive nature of the sound
        zcr = librosa.feature.zero_crossing_rate(y)
        mean_zcr = np.mean(zcr)

        # 4. spectral centroid (indicates the 'center of mass' of the spectrum)
        # relates to the perceived brightness of a sound
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        mean_spectral_centroid = np.mean(spectral_centroid)

        # 5. spectral bandwidth (measures the width of the spectral band around the centroid)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        mean_spectral_bandwidth = np.mean(spectral_bandwidth)

        # 6. root mean square (rms) energy (related to perceived loudness)
        rms = librosa.feature.rms(y=y)
        mean_rms = np.mean(rms)

        # 7. mel-frequency cepstral coefficients (mfccs)
        # capture timbral/textural aspects of the sound; commonly used in speech/music processing
        # we take the first 13 coefficients
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # calculate the mean of each mfcc coefficient across all time frames (axis=1)
        mean_mfccs = np.mean(mfccs, axis=1)

        # 8. chroma features (represent the distribution of energy across 12 pitch classes)
        # useful for analyzing harmony and melody
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        # calculate the mean of each chroma bin across all time frames (axis=1)
        mean_chroma = np.mean(chroma, axis=1)

        # combine all features into a single flat numpy array
        # order: tempo, zcr, centroid, bandwidth, rms, beat_std, mfcc1-13, chroma1-12
        feature_vector = np.concatenate((
            # tempo is often returned as an array, take the first element
            [tempo[0] if isinstance(tempo, np.ndarray) and tempo.size > 0 else tempo],
            [mean_zcr],
            [mean_spectral_centroid],
            [mean_spectral_bandwidth],
            [mean_rms],
            [beat_interval_std], # our calculated beat regularity feature
            mean_mfccs,
            mean_chroma
        ))

        return feature_vector

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# main execution block: runs only when the script is executed directly
if __name__ == '__main__':
    # example usage: demonstrates how to use the extract_features function
    # note: this requires the 'soundfile' library to create a dummy file if needed
    # replace 'example.wav' with an actual .wav file path for real testing
    example_file = 'wav_songs/example.wav'
    print(f"Attempting to extract features from: {example_file}")

    # attempt to create a dummy wav file if the example file doesn't exist
    # this allows the example code to run without needing a pre-existing file
    try:
        import soundfile as sf
        import os
        if not os.path.exists('wav_songs'):
            os.makedirs('wav_songs')
        if not os.path.exists(example_file):
             # create a short silent wav file for basic testing purposes
            sr_test = 22050 # sample rate for the dummy file
            duration_test = 1 # duration in seconds
            # generate an array of zeros representing silence
            silence = np.zeros(int(sr_test * duration_test))
            sf.write(example_file, silence, sr_test)
            print(f"Created dummy file: {example_file}")
    except ImportError:
        print("skipping dummy file creation: 'soundfile' library not installed.")
    except Exception as e_create:
         print(f"could not create dummy file: {e_create}")


    features = extract_features(example_file)

    if features is not None:
        print("Features extracted successfully:")
        print(features)
        print(f"Feature vector shape: {features.shape}")
        # the feature vector should contain 31 elements:
        # 1 (tempo) + 1 (beat_std) + 1 (zcr) + 1 (centroid) + 1 (bandwidth) + 1 (rms) + 13 (mfcc) + 12 (chroma) = 31
        print(f"expected feature vector length: 31")
    else:
        print("Feature extraction failed.")