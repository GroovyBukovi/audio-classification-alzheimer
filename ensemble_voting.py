import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

mfcc = pd.read_csv("top_5_mfcc.csv")
text_features_labeled = pd.read_csv("text_features_labeled.csv")
training_groundtruth = pd.read_csv("cleaned_training_groundtruth.csv")
important_features = pd.read_csv("final_all_features_15_sec.csv")


X_1 = training_groundtruth.drop('dx', axis=1)
X_1 = X_1.drop('adressfname', axis=1)


text_features = text_features_labeled.drop('dx', axis=1)
X_2 = text_features

top_5 = mfcc.sort_values(by='File', ascending=True)
top_5 = top_5.drop('File', axis=1)
top_5 = top_5.drop('dx', axis=1)
# handle categorical values in gender column
X_3 = top_5

y = training_groundtruth.dx

#model1 = svm.SVC(kernel='rbf', gamma=0.1, C=1.5)
model1 = svm.SVC(kernel='rbf', gamma=0.5, C=1.5)
#model2 = LinearDiscriminantAnalysis(solver='eigen', shrinkage=1)
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
from sklearn.metrics import classification_report

accuracy = accuracy_score(y_test, final_prediction)
print("Final Accuracy (Late Fusion with Voting):", accuracy)

f1 = f1_score(final_prediction, y_test, pos_label='Control')
print(f"F1 Score Control: {f1}")
f1 = f1_score(final_prediction, y_test, pos_label='ProbableAD')
print(f"F1 Score ProbableAD: {f1}")

report = classification_report(y_test, final_prediction, output_dict=True)

# Convert to DataFrame for better visualization
df_report = pd.DataFrame(report).transpose()

# Display the F1-score report
print(df_report)  # Replace with display function if needed