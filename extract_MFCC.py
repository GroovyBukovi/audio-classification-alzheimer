import os
import pandas as pd
import numpy as np
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioBasicIO

def extract_mfcc_features_to_csv(audios):
    """
    Extract audio features from a folder of mp3 files and save them to a CSV file.

    Args:
        folder_path (str): Path to the folder containing audio files.
        csv_output_path (str): Path to save the extracted features as a CSV file.
    """
    # List all WAV files in the folder
    print("Extracting MFCC features...")
    audio_files = [os.path.join(audios, f) for f in os.listdir(audios) if f.endswith('.mp3')]

    # Check if there are valid audio files
    if not audio_files:
        raise ValueError("No mp3 files found in the specified folder.")

    all_features = []  # To store features from all files
    file_names = []  # To store corresponding file names

    for file_path in audio_files:
        # Read audio file
        sampling_rate, signal = audioBasicIO.read_audio_file(file_path)
        if signal is None:
            print(f"Error reading file {file_path}, skipping...")
            continue

        # Ensure mono audio
        signal = audioBasicIO.stereo_to_mono(signal)

        """
        Extract mid-term features
        Mid-term features analyze segments of audio data over a longer period than short-term features,
        usually summarizing characteristics like energy, frequency, etc., from a segment of the audio.

        """

        mt_features, _, feature_names = aF.mid_feature_extraction(
            signal, sampling_rate, 45.0 * sampling_rate, 22.5 * sampling_rate, 0.05 * sampling_rate,
                                   0.05 * sampling_rate
        )

        # Average features over windows
        avg_features = np.mean(mt_features, axis=1)
        all_features.append(avg_features)
        file_names.append(os.path.basename(file_path))

    # Combine file names and features into a DataFrame
    mfcc = pd.DataFrame(all_features, columns=feature_names)
    mfcc.insert(0, 'File', file_names)  # Insert file names as the first column

    # Write DataFrame to CSV
    mfcc.to_csv("mfcc_45_sec.csv", index=False)

    print(f"Features saved to mfcc.csv")