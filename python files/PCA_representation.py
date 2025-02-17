import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "/home/droidis/PycharmProjects/projectML/final_important_features_10_15_sec.csv"  # Change to your actual file path
df = pd.read_csv(file_path)

# Convert class labels to numeric for PCA visualization
df["dx"] = df["dx"].map({"Control": 0, "ProbableAD": 1})

# Select only numeric features (excluding target variable)
feature_columns = [col for col in df.columns if col not in ["File", "dx"]]

# Standardize the features (PCA is sensitive to scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_columns])

# Apply PCA (reduce to 2 components for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for visualization
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["Diagnosis"] = df["dx"].map({0: "Control", 1: "ProbableAD"})

# Scatter plot of PCA components
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca["PC1"], y=df_pca["PC2"], hue=df_pca["Diagnosis"], alpha=0.7, palette=["blue", "red"])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection of Features (2D)")
plt.legend(title="Diagnosis")
plt.show()

