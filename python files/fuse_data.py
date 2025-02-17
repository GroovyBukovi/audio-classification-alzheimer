import pandas as pd

def fuse_and_clean_data(mfcc, text_features, groundtruth):
    print("Fusing all the features in a finalized dataframe...")
    # final dataframe preparation
    mfcc = mfcc.sort_values('File')
    mfcc['File'] = mfcc['File'].str.replace('.mp3', '')
    mfcc = mfcc.drop('dx', axis=1)
    text_features['filename'] = text_features['filename'].str.replace('.mp3', '')
    #text_features = text_features.drop('dx', axis=1)

    final_features = mfcc.merge(groundtruth, left_on='File', right_on='adressfname')
    final_features = final_features.merge(text_features, left_on='File', right_on='filename')
    final_features = final_features.drop('adressfname', axis=1)
    final_features = final_features.drop('filename', axis=1)

    final_features.to_csv('final_important_features_10_15_sec.csv', index=False)


mfcc_all = pd.read_csv("/home/droidis/PycharmProjects/projectML/mfcc_15_sec.csv")
mfcc_top = pd.read_csv("/home/droidis/PycharmProjects/projectML/top_10_mfcc.csv")
text_features = pd.read_csv("/home/droidis/PycharmProjects/projectML/text_features.csv")
training_groundtruth = pd.read_csv("/home/droidis/PycharmProjects/projectML/cleaned_training_groundtruth.csv")

fuse_and_clean_data(mfcc_top, text_features, training_groundtruth)