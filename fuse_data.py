import numpy as np

def fuse_and_clean_data(mfcc, text_features, groundtruth):
    print("Fusing all the features in a finalized dataframe...")
    # final dataframe preparation
    mfcc = mfcc.sort_values('File')
    mfcc['File'] = mfcc['File'].str.replace('.mp3', '')
    text_features['filename'] = text_features['filename'].str.replace('.mp3', '')

    final_features = mfcc.merge(groundtruth, left_on='File', right_on='adressfname')
    final_features = final_features.merge(text_features, left_on='File', right_on='filename')
    final_features = final_features.drop('adressfname', axis=1)
    final_features = final_features.drop('filename', axis=1)

    # handle missing education values by assigning the mean. Should i assign 2 different means one for each class?
    mean = final_features[['educ']].mean()
    final_features['educ'] = final_features['educ'].replace(np.nan, float(mean))
    # handle missing mmse values by assigning the mean. Should i assign 2 different means one for each class?
    mean_mmse = training_groundtruth[['mmse']].mean()
    training_groundtruth['mmse'] = training_groundtruth['mmse'].replace(np.nan, float(mean_mmse))

    final_features = final_features.dropna(axis=0)

    # handle categorical values in gender column
    final_features['gender'] = final_features['gender'].map({'male': 0, 'female': 1})

    final_features.to_csv('final_features.csv', index=False)