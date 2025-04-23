import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

def load_data(filepath):
    """Load the dataset from the specified CSV file."""
    df = pd.read_csv(filepath)
    print("Dataset Loaded")
    print("Shape:", df.shape)
    print("Sample Rows:")
    print(df.head(3))
    return df

def preprocess_data(df):
    """Clean the dataset: remove subject prefixes, convert to lowercase, and drop nulls."""
    df['email'] = df['email'].str.replace(r"^Subject:\s*", "", regex=True)
    df['email'] = df['email'].str.lower()
    df.dropna(subset=["email", "type"], inplace=True)
    print("Data Cleaned")
    return df

def vectorize_data(df):
    """Convert email text to TF-IDF feature vectors."""
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(df['email'])
    y = df['type']
    print("TF-IDF vectorization done")
    print("Shape of TF-IDF matrix:", X.shape)
    return X, y, tfidf

def train_and_evaluate_model(X, y):
    """Train a Naive Bayes classifier and evaluate it."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Model trained")
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    return model

def save_model_and_vectorizer(model, vectorizer, model_path="saved_model/model.pkl", vectorizer_path="saved_model/tfidf.pkl"):
    """Save the trained model and TF-IDF vectorizer to disk."""
    os.makedirs("saved_model", exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")

if __name__ == "__main__":
    filepath = os.path.join("data", "combined_emails_with_natural_pii.csv")
    df = load_data(filepath)
    df = preprocess_data(df)
    X, y, tfidf_vectorizer = vectorize_data(df)
    model = train_and_evaluate_model(X, y)
    save_model_and_vectorizer(model, tfidf_vectorizer)
