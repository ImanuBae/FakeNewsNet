import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

print("Training model and saving...")

# Load data
print("Loading data...")
politifact_fake = pd.read_csv('dataset/politifact_fake.csv')
politifact_real = pd.read_csv('dataset/politifact_real.csv')

politifact_fake['label'] = 0
politifact_real['label'] = 1

data = pd.concat([politifact_fake, politifact_real], ignore_index=True)

# Auto-detect text column
if 'title' in data.columns:
    text_column = 'title'
elif 'text' in data.columns:
    text_column = 'text'
else:
    text_column = [col for col in data.columns if col not in ['id', 'label', 'news_url', 'tweet_ids']][0]

data = data.dropna(subset=[text_column])

print(f"Using column: {text_column}")
print(f"Total samples: {len(data)}")

X = data[text_column]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)}")
print(f"Test set: {len(X_test)}")

# Vectorize
print("\nVectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# Test accuracy
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {accuracy:.4f}")

# Save model and vectorizer
print("\nSaving model...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\n" + "="*60)
print("SUCCESS!")
print("="*60)
print("Model saved to 'model.pkl'")
print("Vectorizer saved to 'vectorizer.pkl'")
print(f"Model accuracy: {accuracy:.4f}")
print("\nNext step: Run 'streamlit run app.py'")
print("="*60)