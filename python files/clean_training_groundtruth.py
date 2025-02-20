import pandas as pd

training_groundtruth = pd.read_csv("/home/droidis/PycharmProjects/projectML/training-groundtruth.csv")
# Compute class-specific means for 'educ' and fill null cells
class_means_educ = training_groundtruth.groupby('dx')['educ'].transform(lambda x: x.fillna(x.mean()))
training_groundtruth['educ'] = class_means_educ

# Drop remaining NaN values (if any)
training_groundtruth = training_groundtruth.dropna(axis=0)
# Encode categorical values in 'gender'
training_groundtruth['gender'] = training_groundtruth['gender'].map({'male': 0, 'female': 1})

training_groundtruth = training_groundtruth.drop('mmse', axis=1)

training_groundtruth.to_csv('cleaned_training_groundtruth.csv', index=False)