# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# Import the preprocessing function
from utils import preprocess_text

print("Starting model training process...")

# Load and Combine Data
data_path = './data/'
df_list = []

try:
    # Load new REAL news
    df_real = pd.read_csv(os.path.join(data_path, 'True.csv'))
    df_real['label'] = 1  # 1 for Real
    df_list.append(df_real[['text', 'label']])
    print("Loaded True.csv")

    # Load new FAKE news
    df_fake = pd.read_csv(os.path.join(data_path, 'Fake.csv'))
    df_fake['label'] = 0  # 0 for Fake
    df_list.append(df_fake[['text', 'label']])
    print("Loaded Fake.csv")

except FileNotFoundError:
    print("\n--- ERROR ---")
    print("Could not find 'True.csv' or 'Fake.csv' in the 'data/' folder.")
    print("Please download the dataset from Kaggle and place it there.")
    print("----------------\n")
    exit()

# Combine all dataframes
df = pd.concat(df_list, ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Loaded and combined {len(df)} articles.")

# Preprocessing 

# Handle potential missing (NaN) values by filling them with empty strings
df['text'] = df['text'].fillna('')

# NOTE: We only have a 'text' column now, not 'title'.
df['content'] = df['text']

# Drop rows where content is just an empty space
df = df[df['content'].str.strip().astype(bool)]

print("Applying text preprocessing... (This will take several minutes)")

df['processed_content'] = df['content'].apply(preprocess_text)

print("Preprocessing complete.")

# Feature Engineering (TF-IDF)
X = df['processed_content']
y = df['label']

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

print("Fitting TF-IDF vectorizer...")
X_tfidf = vectorizer.fit_transform(X)

print("TF-IDF vectorization complete.")

# Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# Model Training (Random Forest)
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,  # 100 trees
    random_state=42,
    n_jobs=-1         # Use all available CPU cores
)
model.fit(X_train, y_train)
print("Model training complete.")

# Model Evaluation 
print("Evaluating model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n--- Model Evaluation Results ---")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
# Target names are important for the report
report = classification_report(y_test, y_pred, target_names=['Fake (0)', 'Real (1)'])
print(report)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("---------------------------------")


# Save the Model and Vectorizer 
saved_model_path = './saved_models/'
if not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)

pickle.dump(model, open(os.path.join(saved_model_path, 'model.pkl'), 'wb'))
pickle.dump(vectorizer, open(os.path.join(saved_model_path, 'vectorizer.pkl'), 'wb'))

print(f"\nModel and vectorizer saved to {saved_model_path}")
print("Training process finished.")