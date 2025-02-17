import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace with your actual file path)
file_path = "/home/droidis/PycharmProjects/projectML/enhanced_training_groundtruth.csv"
df = pd.read_csv(file_path)

# Selecting relevant columns
columns_of_interest = ['age', 'gender', 'educ', 'age_educ_ratio', 'dx']
df_filtered = df[columns_of_interest].copy()

# Encode 'dx' as a numeric variable if it is categorical
dx_mapping = {'Control': 0, 'ProbableAD': 1}
df_filtered['dx'] = df_filtered['dx'].map(dx_mapping)

# Compute the correlation matrix
correlation_matrix = df_filtered.corr()

# Set up the matplotlib figure
plt.figure(figsize=(8, 6))

# Draw the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Title and display
plt.title("Correlation Matrix Heatmap")
plt.show()