import pandas as pd

# Load the dataset

df = pd.read_csv("/home/droidis/PycharmProjects/projectML/cleaned_training_groundtruth.csv")

# Ensure the necessary columns exist
if 'age' in df.columns and 'educ' in df.columns:
    # Calculate the enhancement column
    df['age_educ_ratio'] = df['age'] / df['educ']
else:
    missing_columns = [col for col in ['age', 'educ'] if col not in df.columns]
    raise ValueError(f"Missing required columns: {missing_columns}")

# Save the enhanced dataset
output_file_path = "../enhanced_training_groundtruth.csv"
df.to_csv(output_file_path, index=False)

# Display the first few rows to verify
print(df.head())

print(f"Enhanced dataset saved as: {output_file_path}")
