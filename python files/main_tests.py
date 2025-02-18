import csv
import time
from datetime import datetime
from fit_predict_accuracy import fit_predict_accuracy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioTrainTest as aT
from scipy.stats import mode
from sklearn.model_selection import KFold, cross_val_score
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import assemblyai as aai
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder



#audios = "train"
mfcc = pd.read_csv("/home/droidis/PycharmProjects/projectML/top_5_mfcc.csv")
text_features_labeled = pd.read_csv("/home/droidis/PycharmProjects/projectML/text_features_labeled.csv")
training_groundtruth = pd.read_csv("/home/droidis/PycharmProjects/projectML/enhanced_training_groundtruth.csv")
important_features = pd.read_csv("/home/droidis/PycharmProjects/projectML/final_important_features_5_15_sec.csv")

################### MFCC ONLY ################### model = svm.SVC(kernel='rbf', gamma=0.07, C=2.5)
#X = mfcc.drop('dx', axis=1)  # Features
#X = X.drop('File', axis=1)
#y = mfcc.dx

############ TOP 5 MFCC ####################  model = svm.SVC(kernel='rbf', gamma=0.9, C=0.8)

#X = top_5.drop('dx', axis=1)  # Features
#X = X.drop('File', axis=1)
#y = top_5.dx

################### TEXT FEATURES ONLY ################  model = LogisticRegression(C=0.5, penalty='l2', max_iter=200, tol=1e-4)
## LinearDiscriminantAnalysis(solver='eigen', shrinkage=1) #73% for initial data, normalized and standardized
#X = text_features_labeled.drop('dx', axis=1)  # Features
#y = text_features_labeled.dx

################### TRAINING GROUND TRUTH ONLY ################
###### SVM???? ######## model = svm.SVC(kernel='rbf', gamma=0.5, C=1.5)

# Feature selection
X = training_groundtruth.drop(['dx', 'adressfname'], axis=1)  # Features
y = training_groundtruth['dx']

scaler = StandardScaler()

# Standardize the DataFrame
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

######################## NORMALIZE-STANDARDISE ##########################

# normalise
#scaler = MinMaxScaler()

# standardize
scaler = StandardScaler()

# Standardize the DataFrame
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

########## MODEL WITH ALL FEATURES FUSED ############## model = svm.SVC(kernel='rbf', gamma=0.007, C=1, class_weight={'Control': 1, 'ProbableAD': 4})
#important_features['dx'] = important_features['dx'].map({'Control': 0, 'ProbableAD': 1})
'''X = important_features

X = X.drop('dx', axis=1)  # Features
#X = X.drop('dx_y', axis=1)  # Features
X = X.drop('File', axis=1)

y = important_features.dx # Target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Assuming y contains the original class labels
print(le.classes_)  # This will show the order of classes
# normalise
#scaler = MinMaxScaler()


# standardize
scaler = StandardScaler()

# Standardize the DataFrame
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

start_time = datetime.now()
kf = KFold(n_splits=10, shuffle=True, random_state=42)
model = svm.SVC(kernel='rbf', gamma=0.009, C=2, class_weight={'Control': 1, 'ProbableAD': 1.3})

y_pred = cross_val_predict(model, X, y, cv=kf)

cnf_matrix = metrics.confusion_matrix(y, y_pred)

# Print confusion matrix
print("Confusion Matrix:\n", cnf_matrix)
f1 = f1_score(y_pred, y, pos_label='Control')
print(f"F1 Score Control: {f1}")
f1 = f1_score(y_pred, y, pos_label='ProbableAD')
print(f"F1 Score ProbableAD: {f1}")

cv_scores = cross_val_score(model, X, y, cv=kf)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Accuracy: {np.mean(cv_scores):.4f}")

end_time = datetime.now()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time}")'''
########## ENSEMBLE VOTING APPROACH ##############
'''
# standardize
scaler = StandardScaler()

X_1 = training_groundtruth.drop('dx', axis=1)
X_1 = X_1.drop('adressfname', axis=1)
X_1 = pd.DataFrame(scaler.fit_transform(X_1), columns=X_1.columns)

text_features = text_features_labeled.drop('dx', axis=1)
X_2 = text_features
X_2 = pd.DataFrame(scaler.fit_transform(X_2), columns=X_2.columns)


top_5 = mfcc.sort_values(by='File', ascending=True)
top_5 = top_5.drop('File', axis=1)
top_5 = top_5.drop('dx', axis=1)
# handle categorical values in gender column
X_3 = top_5
X_2 = pd.DataFrame(scaler.fit_transform(X_2), columns=X_2.columns)


y = training_groundtruth.dx

model1 = LogisticRegression(C=0.1, penalty='l2', max_iter=100, class_weight={'Control': 1, 'ProbableAD': 1.24})
#model1 = svm.SVC(kernel='rbf', C=0.05)
#model1 = svm.SVC(kernel='rbf', gamma=0.5, C=1.5)
model2 = LinearDiscriminantAnalysis(solver='eigen', shrinkage=1)
#model2 = LogisticRegression(C=0.5, penalty='l2', max_iter=100, tol=1e-4)
#model3 = svm.SVC(kernel='rbf', gamma=0.9, C=0.8)
model3 = svm.SVC(kernel='rbf',  C=0.8)

print(X_1.head())
print(X_2.head())
print(X_3.head())


X_train1, X_test1, y_train, y_test = train_test_split(X_1, y, test_size=0.3, random_state=16)
X_train2, X_test2, _, _  = train_test_split(X_2, y, test_size=0.3, random_state=16)
X_train3, X_test3, _, _  = train_test_split(X_3, y, test_size=0.3, random_state=16)
start_time = datetime.now()
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
from sklearn.metrics import classification_report

accuracy = accuracy_score(y_test, final_prediction)
print("Final Accuracy (Late Fusion with Voting):", accuracy)

f1 = f1_score(final_prediction, y_test, pos_label='Control')
print(f"F1 Score Control: {f1}")
f1 = f1_score(final_prediction, y_test, pos_label='ProbableAD')
print(f"F1 Score ProbableAD: {f1}")
cnf_matrix = metrics.confusion_matrix(y_test, final_prediction)
print(cnf_matrix)
report = classification_report(y_test, final_prediction, output_dict=True)

# Convert to DataFrame for better visualization
df_report = pd.DataFrame(report).transpose()

# Display the F1-score report
print(df_report)  # Replace with display function if needed
end_time = datetime.now()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time}")
'''
##################################################################333
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=16)

model = LogisticRegression(C=0.1, penalty='l2', max_iter=100, class_weight={'Control': 1, 'ProbableAD': 1.24})
#model = XGBClassifier(scale_pos_weight=3, learning_rate=1, max_depth=3)
#model = svm.SVC(kernel='rbf', gamma=0.01, C=3, class_weight={'Control': 1, 'ProbableAD': 1.3})
#model = svm.SVC(kernel='rbf', gamma=0.009, C=2, class_weight={'Control': 1, 'ProbableAD': 1.3}) # all top 5
#model = LogisticRegression(C=0.1, penalty='l2', max_iter=100, class_weight={'Control': 1, 'ProbableAD': 1.24}) # groundtruth
#model = KNeighborsClassifier(n_neighbors=3, weights='uniform') # for normalised data it gives 76% accuracy, 92% for non initial and 76% for standardized. Test different amount of neighbours
#model = DecisionTreeClassifier(max_depth=5,             # limit the tree depth
                               #min_samples_split=3,    # require at least 10 samples to split a node
                               #min_samples_leaf=10,      # require at least 5 samples per leaf
                               #max_features="sqrt",     # use square root of the features for splits
                               #random_state=7,
                               #class_weight={'Control': 1, 'ProbableAD': 2})          # for reproducibility
#model = RandomForestClassifier(n_estimators=3,           # Reduce trees to control complexity
        #max_depth=10,              # Prevent deep, complex trees
        #min_samples_split=5,     # Require at least 10 samples to split a node
        #min_samples_leaf=10,       # Require at least 5 samples per leaf
        #max_features="sqrt",      # Use only a subset of features per tree
        #random_state=10,
                               #class_weight={'Control': 1, 'ProbableAD': 2})
#model = GaussianNB(var_smoothing=1e-1) # 81% for normalized and standardized, 83% for initial data
#model = LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.7) #73% for initial data, normalized and standardized




fit_predict_accuracy(model, X_train, X_test, y_train, y_test)


# LDA because assumes linearity which indeed exists, no standardization because tends to overfit.??
#final_all_features_15_sec - model = LinearDiscriminantAnalysis(solver='eigen', shrinkage=1)
# final_important_features_5_15_sec - model = LinearDiscriminantAnalysis(solver='eigen', shrinkage=1)


# SVM because works better with high dimensional data
# final_important_features_5_15_sec - model = svm.SVC(kernel='rbf', gamma=0.15, C=2)
# final_all_features_15_sec - model = svm.SVC(kernel='rbf', gamma=0.13, C=2)


######################## ENSEMBE K-FOLD #####################################
'''from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd

# Initialize scaler
scaler = StandardScaler()

# Standardize feature sets
X_1 = training_groundtruth.drop(['dx', 'adressfname'], axis=1)
X_1 = pd.DataFrame(scaler.fit_transform(X_1), columns=X_1.columns)

X_2 = text_features_labeled.drop('dx', axis=1)
X_2 = pd.DataFrame(scaler.fit_transform(X_2), columns=X_2.columns)

X_3 = mfcc.sort_values(by='File', ascending=True).drop(['File', 'dx'], axis=1)
X_3 = pd.DataFrame(scaler.fit_transform(X_3), columns=X_3.columns)

y = training_groundtruth.dx  # Target variable

# Define models
#model1 = SVC(kernel='rbf', gamma=0.1, C=0.07, class_weight={'Control': 1, 'ProbableAD': 1.1})
#model2 = LogisticRegression(C=0.5, penalty='l2', max_iter=200, tol=1e-4)
#model2 = LinearDiscriminantAnalysis(solver='eigen', shrinkage=1)
#model3 = SVC(kernel='rbf', gamma=0.9, C=0.8, class_weight={'Control': 1, 'ProbableAD': 1.2})

#model1 = LogisticRegression(C=0.0001, penalty='l2', max_iter=50, tol=1e-0)
model1 = svm.SVC(kernel='rbf', C=0.05)
#model1 = svm.SVC(kernel='rbf', gamma=0.5, C=1.5)
model2 = LinearDiscriminantAnalysis(solver='eigen', shrinkage=1)
#model2 = LogisticRegression(C=0.5, penalty='l2', max_iter=100, tol=1e-4)
#model3 = svm.SVC(kernel='rbf', gamma=0.9, C=0.8)
model3 = svm.SVC(kernel='rbf',  C=0.8)'''

'''
#model1 = LogisticRegression(C=0.17, penalty='l2', max_iter=100, class_weight={'Control': 1, 'ProbableAD': 1.3})
#model1 = svm.SVC(kernel='rbf', C=0.05)
#model1 = svm.SVC(kernel='rbf', gamma=0.5, C=1.5)
model2 = LinearDiscriminantAnalysis(solver='eigen', shrinkage=1)
#model2 = LogisticRegression(C=0.5, penalty='l2', max_iter=100, tol=1e-4)
#model3 = svm.SVC(kernel='rbf', gamma=0.9, C=0.8)
model3 = svm.SVC(kernel='rbf',  C=0.8)

# Set up K-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists for storing metrics
accuracy_scores = []
f1_scores_control = []
f1_scores_probableAD = []
conf_matrices = []

# Perform K-Fold CV
for train_index, test_index in kf.split(X_1, y):
    # Split data for each fold
    X_train1, X_test1 = X_1.iloc[train_index], X_1.iloc[test_index]
    X_train2, X_test2 = X_2.iloc[train_index], X_2.iloc[test_index]
    X_train3, X_test3 = X_3.iloc[train_index], X_3.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train models
    fit_1 = model1.fit(X_train1, y_train)
    fit_2 = model2.fit(X_train2, y_train)
    fit_3 = model3.fit(X_train3, y_train)

    # Get predictions
    pred_1 = model1.predict(X_test1)
    pred_2 = model2.predict(X_test2)
    pred_3 = model3.predict(X_test3)

    # Majority voting
    predictions = np.array([pred_1, pred_2, pred_3])  # Shape (3, n_samples)
    df_preds = pd.DataFrame(predictions.T)  # Transpose
    final_prediction = df_preds.mode(axis=1)[0].values  # Majority vote

    # Compute metrics
    accuracy = accuracy_score(y_test, final_prediction)
    f1_control = f1_score(y_test, final_prediction, pos_label='Control')
    f1_probableAD = f1_score(y_test, final_prediction, pos_label='ProbableAD')
    cnf_matrix = confusion_matrix(y_test, final_prediction)

    # Store results
    accuracy_scores.append(accuracy)
    f1_scores_control.append(f1_control)
    f1_scores_probableAD.append(f1_probableAD)
    conf_matrices.append(cnf_matrix)

# Final aggregated results
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Mean F1 Score Control: {np.mean(f1_scores_control):.4f}")
print(f"Mean F1 Score ProbableAD: {np.mean(f1_scores_probableAD):.4f}")

# Summed confusion matrix across all folds
final_conf_matrix = np.sum(conf_matrices, axis=0)
print("Final Confusion Matrix:\n", final_conf_matrix)

# Classification report
final_report = classification_report(y_test, final_prediction, output_dict=True)
df_final_report = pd.DataFrame(final_report).transpose()
print(df_final_report)'''
