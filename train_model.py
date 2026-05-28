import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true])

# Shuffle data
data = data.sample(frac=1, random_state=42)

# Input and Output
x = data["text"]
y = data["label"]

# Convert text to vectors
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

x_vectorized = vectorizer.fit_transform(x)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x_vectorized,
    y,
    test_size=0.25,
    random_state=42
)

# Better Model
model = PassiveAggressiveClassifier(max_iter=1000)

# Train
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Accuracy
score = accuracy_score(y_test, y_pred)

print("Accuracy:", score)

# Save model
pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model Saved Successfully")