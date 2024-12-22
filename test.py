import os
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioTrainTest as aT
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score



def split_data(features, labels, test_size=0.2):
    return train_test_split(features, labels, test_size=test_size, random_state=42)

'''aT.extract_features_and_train(["/home/droidis/Desktop/sample/1", "/home/droidis/Desktop/sample/2"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "svmSMtemp", False)
#aT.file_classification("data/doremi.wav", "svmSMtemp","svm")'''


def extract_features_to_csv(folder_path, csv_output_path):
    """
    Extract audio features from a folder of WAV files and save them to a CSV file.

    Args:
        folder_path (str): Path to the folder containing audio files.
        csv_output_path (str): Path to save the extracted features as a CSV file.
    """
    # List all WAV files in the folder
    audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mp3')]

    # Check if there are valid audio files
    if not audio_files:
        raise ValueError("No WAV files found in the specified folder.")

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

        # Extract mid-term features
        mt_features, _, feature_names = aF.mid_feature_extraction(
            signal, sampling_rate, 1.0 * sampling_rate, 0.5 * sampling_rate, 0.05 * sampling_rate, 0.05 * sampling_rate
        )

        # Average features over windows
        avg_features = np.mean(mt_features, axis=1)
        all_features.append(avg_features)
        file_names.append(os.path.basename(file_path))

    # Combine features and file names into a single array
    feature_data = np.array(all_features)
    header = ["File"] + feature_names

    # Write to CSV
    with open(csv_output_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)  # Write header
        for i, row in enumerate(feature_data):
            csvwriter.writerow([file_names[i]] + row.tolist())

    print(f"Features saved to {csv_output_path}")


# Example usage
#extract_features_to_csv("/home/droidis/Desktop/train", "/home/droidis/Desktop/MFCC.csv")

df1 = pd.read_csv('MFCC.csv')
df2 = pd.read_csv('training-groundtruth.csv')

df1=df1.sort_values('File')
df1['File'] = df1['File'].str.replace('.mp3','')

df_final = df1.merge(df2, left_on = 'File', right_on = 'adressfname')
df_final = df_final.drop('adressfname', axis=1)

mean= df_final[['educ']].mean()
df_final['educ'] = df_final['educ'].replace(np.nan,float(mean))
df_final=df_final.dropna(axis=0)

X = df_final.drop('dx',axis=1 ) # Features
X = X.drop('File',axis=1 )
X = X.drop('gender',axis=1 )


y = df_final.dx # Target variable



#print(X.isnull().sum())

#print(X.to_string())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred=model.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
accuracy = accuracy_score(y_pred, y_test)
print(accuracy)



