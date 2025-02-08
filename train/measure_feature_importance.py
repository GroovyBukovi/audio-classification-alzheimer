import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mfcc = pd.read_csv("mfcc_15_sec.csv")
X = mfcc.drop('dx', axis=1)  # Features
X = X.drop('File', axis=1)
y = mfcc.dx

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df.to_string())