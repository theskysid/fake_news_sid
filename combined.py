import pandas as pd
from sklearn.utils import shuffle
import os

# Step 1: Load the CSV files
fake_df = pd.read_csv('Data/Fake.csv')
true_df = pd.read_csv('Data/True.csv')

# Step 2: Add a 'label' column to each
fake_df['label'] = 0  # 0 for Fake
true_df['label'] = 1  # 1 for Real

# Step 3: Combine the datasets
combined_df = pd.concat([fake_df, true_df], ignore_index=True)

# Step 4: Shuffle the data
combined_df = shuffle(combined_df, random_state=42)

# Optional: Reset index after shuffle
combined_df.reset_index(drop=True, inplace=True)

# Step 5: Save to a new CSV for future use
output_path = 'Data/combined_news.csv'
combined_df.to_csv(output_path, index=False)

print(f"✅ Preprocessing complete. Saved to: {output_path}")
print(combined_df.head())
