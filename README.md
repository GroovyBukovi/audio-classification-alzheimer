# audio-classification-alzheimer
ML project for binary classification of patient recordings with probable Alzheimer's disease.


ML Project by Emmanuellam2408 and GroovyBukovi exploring different methods in audio binary classification. Specifically, the audio corpus consists of potential Alzheimer's disease (AD) patients' recordings and is relatively balanced between 'healthy' and 'potentialAD' individuals.

Each audio file contains a conversation between the interviewer (healthcare professional) and the interviewee (patient), raising the need for diarization and audio splitting to specifically train, validate, and test our model using patient-only audio.

Additionally, along with the audio files, there is a dataset containing useful metadata such as gender, age, cognitive-ability-related features, and, of course, the labels for each instance (healthy, potentialAD).

The project aims to unify these features with the MFCCs (Mel-Frequency Cepstral Coefficients) extracted from the audios and train a classification model based on them.

Process:

Initial Classification:
Perform a simple classification using pyAudioAnalysis, aggregating all features and without removing the interviewer from the audio to establish a plain benchmark for comparison.

Diarization and Transcription:
Utilize WhisperX to obtain necessary diarization and transcription information.

Feature Extraction from WhisperX:
Extract features such as the variance and frequency of words used (indicative of AD), and the duration of utterances.

Audio Splitting:
Split the audio files based on diarization information, isolating interviewee-only (patient) audio. Extract MFCCs from these isolated files.

Feature Fusion and Dataset Preparation:
Combine all gathered features into a single dataset. Split the dataset into training, validation, and testing subsets, and train the model using a variety of classification algorithms to identify the most efficient one.

Model Evaluation:
Evaluate the models using metrics such as F1 score, confusion matrix, AUC/ROC, and computational performance to determine the best model.

Final Documentation:
Create a PDF presentation detailing the entire process, including final thoughts, obstacles faced, and key discoveries.

To Be Updated.
