import csv
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioTrainTest as aT
from scipy.stats import mode
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
import assemblyai as aai
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#from diarization_and_feature_extraction_from_text import text_features

#aT.extract_features_and_train(["train"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "svmSMtemp", False)
#aT.file_classification("data/doremi.wav", "svmSMtemp","svm")'''


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
            signal, sampling_rate, 45.0 * sampling_rate, 22.5 * sampling_rate, 0.05 * sampling_rate, 0.05 * sampling_rate
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
    return mfcc

def diarization_and_feature_extraction_from_text(audios):
    '''print("Diarizing and extracting features from text...")
    text_features = pd.DataFrame(columns=['filename', 'Word Variance', 'Hapax Legomena'])
    # Replace with your API key
    aai.settings.api_key = "262ec24608c0442483768e3004d70d53"
    config = aai.TranscriptionConfig(speaker_labels=True)

    directory = os.fsencode(audios)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".mp3"):
            url = audios + '/' + filename
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(
                url,
                config=config
            )

            for utterance in transcript.utterances:
                print(f"Speaker {utterance.speaker}: {utterance.text}")
                text = utterance.text

        tokens = word_tokenize(text.lower())

        # Frequency distribution
        fdist = FreqDist(tokens)
        word_variance = len(fdist) / len(tokens)  # Lexical richnes
        hapax_legomena = len([word for word, freq in fdist.items() if freq == 1])
        text_features = text_features._append(
            {'filename': filename, 'Word Variance': word_variance, 'Hapax Legomena': hapax_legomena}, ignore_index=True)

    text_features = text_features.sort_values('filename')
    text_features.to_csv('text_features.csv', index=False)
    return text_features'''



    # Download NLTK tokenizer if missing
    nltk.download('punkt')
    print("Diarizing and extracting features from text...")

    # ✅ Creating DataFrame just like before
    text_features = pd.DataFrame(columns=['filename', 'Word Variance', 'Hapax Legomena'])

    # ✅ Replace with your API key
    aai.settings.api_key = "262ec24608c0442483768e3004d70d53"
    config = aai.TranscriptionConfig(speaker_labels=True)

    directory = os.fsencode(audios)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        if filename.endswith(".mp3"):
            url = os.path.join(audios, filename)  # Ensure correct file path
            transcriber = aai.Transcriber()

            try:
                transcript = transcriber.transcribe(url, config=config)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue  # Skip file if transcription fails

            # ✅ Combine all speakers' utterances into one text block per audio file
            full_text = " ".join([utterance.text for utterance in transcript.utterances])

            if not full_text.strip():
                print(f"No speech detected in {filename}, skipping...")
                continue  # Skip empty transcriptions

            tokens = word_tokenize(full_text.lower())  # Tokenize text

            if not tokens:
                print(f"No valid words in {filename}, skipping...")
                continue  # Skip if no tokens

            # ✅ Compute features on the combined text
            fdist = FreqDist(tokens)
            word_variance = len(fdist) / len(tokens)  # Lexical richness
            hapax_legomena = sum(1 for freq in fdist.values() if freq == 1)  # Count words appearing only once

            # ✅ Append row to DataFrame just like before
            text_features = text_features._append(
                {'filename': filename, 'Word Variance': word_variance, 'Hapax Legomena': hapax_legomena},
                ignore_index=True
            )

    # ✅ Sorting and saving to CSV just like before
    text_features = text_features.sort_values('filename')
    text_features.to_csv('text_features.csv', index=False)

    return text_features  # ✅ Returning DataFrame just like before



def fuse_and_clean_data(mfcc, text_features, groundtruth):
    print("Fusing all the features in a finalized dataframe...")
    # final dataframe preparation
    mfcc = mfcc.sort_values('File')
    mfcc = mfcc.drop('dx', axis=1)
    mfcc['File'] = mfcc['File'].str.replace('.mp3', '')
    text_features['filename'] = text_features['filename'].str.replace('.mp3', '')

    final_features = mfcc.merge(groundtruth, left_on='File', right_on='adressfname')
    final_features = final_features.merge(text_features, left_on='File', right_on='filename')
    final_features = final_features.drop('adressfname', axis=1)
    final_features = final_features.drop('filename', axis=1)

    final_features.to_csv('final_important_features_20_15_sec.csv', index=False)


def fuse_and_clean_data_no_text(mfcc, groundtruth):
    print("Fusing all the features in a finalized dataframe...")
    # final dataframe preparation
    mfcc = mfcc.sort_values('File')
    mfcc['File'] = mfcc['File'].str.replace('.mp3', '')


    final_features = mfcc.merge(groundtruth, left_on='File', right_on='adressfname')
    final_features = final_features.drop('adressfname', axis=1)

    # handle missing education values by assigning the mean. Should i assign 2 different means one for each class?
    mean = final_features[['educ']].mean()
    final_features['educ'] = final_features['educ'].replace(np.nan, float(mean))

    final_features = final_features.dropna(axis=0)

    # handle categorical values in gender column
    final_features['gender'] = final_features['gender'].map({'male': 0, 'female': 1})

    final_features.to_csv('final_features.csv', index=False)

    return final_features

def fit_predict_accuracy(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=["Control", "ProbableAD"])
    cm_display.plot()
    #plt.show()
    print(f"CNF Matrix: \n {cnf_matrix}")
    accuracy = accuracy_score(y_pred, y_test)
    print(f"Accuracy: {accuracy}")
    f1 = f1_score(y_pred, y_test, pos_label='Control')
    print(f"F1 Score Control: {f1}")
    f1 = f1_score(y_pred, y_test, pos_label='ProbableAD')
    print(f"F1 Score ProbableAD: {f1}")

def main(audios,model):
    start_time = time.time()
    mfcc = extract_mfcc_features_to_csv(audios)
    text_features = diarization_and_feature_extraction_from_text(audios)
    training_groundtruth = pd.read_csv('training-groundtruth.csv')

    print("Finalizing data and splitting in test and training corpuses...")
    #final dataframe preparation
    final_features = fuse_and_clean_data(mfcc,text_features, training_groundtruth)

    X = final_features.drop('dx', axis=1)  # Features
    X = X.drop('File', axis=1)
    y = final_features.dx  # Target variable

    # Normalize the DataFrame
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime for data preparation: {runtime} seconds")
    fit_predict_accuracy(model, X_train, X_test, y_train, y_test)






# Example usage
#extract_features_to_csv("train", "mfcc_45_sec")
#extract_features_to_csv("/home/droidis/Desktop/train_edited", "MFCC_edited.csv")


audios = "train"
#main(audios, model)

mfcc = pd.read_csv("mfcc_30_sec.csv")
text_features_labeled = pd.read_csv("text_features_labeled.csv")
training_groundtruth = pd.read_csv("cleaned_training_groundtruth.csv")
important_features = pd.read_csv("final_important_features_5_15_sec.csv")

################### MFCC ONLY ###################
#X = mfcc_labeled.drop('dx', axis=1)  # Features
#X = X.drop('File', axis=1)
#y = mfcc_labeled.dx



############ TOP 20 MFCC ####################  model = svm.SVC(kernel='rbf', gamma=0.9, C=0.8)

#X = top_5.drop('dx', axis=1)  # Features
#X = X.drop('File', axis=1)
#y = top_5.dx

################### TEXT FEATURES ONLY ################  model = LogisticRegression(C=0.5, penalty='l2', max_iter=200, tol=1e-4)
#X = text_features_labeled.drop('dx', axis=1)  # Features
#y = text_features_labeled.dx

################### TRAINING GROUND TRUTH ONLY ################
###### SVM???? ######## model = svm.SVC(kernel='rbf', gamma=0.1, C=0.4)

# Feature selection
#X = training_groundtruth.drop(['dx', 'adressfname'], axis=1)  # Features
#y = training_groundtruth['dx']

######################## NORMALIZE-STANDARDISE ##########################

# normalise
#scaler = MinMaxScaler()

# standardize
#scaler = StandardScaler()

# Standardize the DataFrame
#X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

########## MODEL WITH ALL FEATURES FUSED ##############
'''
X = final_features
#X = final_features.drop('dx', axis=1)  # Features
#X = X.drop('dx', axis=1)  # Features
X = X.drop('File', axis=1)

y = training_groundtruth.dx # Target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Assuming y contains the original class labels
print(le.classes_)  # This will show the order of classes
# normalise
#scaler = MinMaxScaler()


# standardize
scaler = StandardScaler()

# Standardize the DataFrame
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

'''
########## ENSEMBLE VOTING APPROACH ##############
'''
mean_educ = training_groundtruth[['educ']].mean()
training_groundtruth['educ'] = training_groundtruth['educ'].replace(np.nan, float(mean_educ))


mean_mmse = training_groundtruth[['mmse']].mean()
training_groundtruth['mmse'] = training_groundtruth['mmse'].replace(np.nan, float(mean_mmse))
training_groundtruth = training_groundtruth.drop('adressfname', axis=1)

#training_groundtruth = training_groundtruth.dropna(axis=0)


# handle categorical values in gender column
training_groundtruth['gender'] = training_groundtruth['gender'].map({'male': 0, 'female': 1})

X_1 = training_groundtruth.drop('dx', axis=1)

text_features = text_features.drop('filename', axis=1)

#text_features = text_features.dropna(axis=0)


X_2 = text_features

top_5= top_5.sort_values(by='File', ascending=True)
top_5 = top_5.drop('File', axis=1)
top_5 = top_5.drop('dx', axis=1)
# handle categorical values in gender column

X_3 = top_5

y = training_groundtruth.dx

model1 = svm.SVC(kernel='rbf', gamma=0.1, C=1.5)
model2 = LogisticRegression(C=0.5, penalty='l2', max_iter=200, tol=1e-4)
model3 = svm.SVC(kernel='rbf', gamma=0.9, C=0.8)

print(X_1.head())
print(X_2.head())
print(X_3.head())


X_train1, X_test1, y_train, y_test = train_test_split(X_1, y, test_size=0.3, random_state=16)
X_train2, X_test2, _, _  = train_test_split(X_2, y, test_size=0.3, random_state=16)
X_train3, X_test3, _, _  = train_test_split(X_3, y, test_size=0.3, random_state=16)

fit_1=model1.fit(X_train1, y_train)
fit_2=model2.fit(X_train2, y_train)
fit_3=model3.fit(X_train3, y_train)

pred_1=model1.predict(X_test1)
pred_2=model2.predict(X_test2)
pred_3=model3.predict(X_test3)

# Combine predictions and apply majority voting
predictions = np.array([pred_1, pred_2, pred_3])  # Shape (3, n_samples)
df_preds = pd.DataFrame(predictions.T)  # Transpose to get samples as rows
final_prediction = df_preds.mode(axis=1)[0].values  # Select first mode if tie


accuracy = accuracy_score(y_test, final_prediction)
print("Final Accuracy (Late Fusion with Voting):", accuracy)
'''
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=16)

#model = svm.SVC(kernel='rbf', gamma=0.5, C=2)
#model = LogisticRegression(C=0.1, penalty='l2', max_iter=200, tol=1e-4)
#model = KNeighborsClassifier(n_neighbors=10) # for normalised data it gives 76% accuracy, 92% for non initial and 76% for standardized. Test different amount of neighbours
#model = DecisionTreeClassifier(max_depth=5,             # limit the tree depth
                               #min_samples_split=2,    # require at least 10 samples to split a node
                               #min_samples_leaf=5,      # require at least 5 samples per leaf
                               #max_features="sqrt",     # use square root of the features for splits
                               #random_state=4)          # for reproducibility
#model = RandomForestClassifier(n_estimators=5,           # Reduce trees to control complexity
    #max_depth=2,              # Prevent deep, complex trees
    #min_samples_split=3,     # Require at least 10 samples to split a node
    #min_samples_leaf=5,       # Require at least 5 samples per leaf
    #max_features="sqrt",      # Use only a subset of features per tree
    #random_state=42)
#model = GaussianNB(var_smoothing=1e-1) # 81% for normalized and standardized, 83% for initial data
#model = LinearDiscriminantAnalysis(solver='eigen', shrinkage=1) #73% for initial data, normalized and standardized




#fit_predict_accuracy(model, X_train, X_test, y_train, y_test)

'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df.to_string())'''



fuse_and_clean_data(top_20, text_features, training_groundtruth)
