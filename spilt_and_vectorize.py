import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Load combined data
df = pd.read_csv('Data/combined_news.csv')

# Split into input (X) and output (y)
X = df['title']
y = df['label']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save the vectorized data and the vectorizer for later use
os.makedirs('Data/processed', exist_ok=True)
joblib.dump(X_train_tfidf, 'Data/processed/X_train_tfidf.pkl')
joblib.dump(X_test_tfidf, 'Data/processed/X_test_tfidf.pkl')
joblib.dump(y_train, 'Data/processed/y_train.pkl')
joblib.dump(y_test, 'Data/processed/y_test.pkl')
joblib.dump(vectorizer, 'Data/processed/tfidf_vectorizer.pkl')

print("✅ Data vectorized and saved in 'Data/processed/' folder.")
