import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Labels
fake["label"] = 0
true["label"] = 1

# Combine
data = pd.concat([fake, true], ignore_index=True)

# Shuffle
data = data.sample(frac=1, random_state=42)

# Features and target
x = data["text"]
y = data["label"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english")

x = vectorizer.fit_transform(x)

# Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Model
model = MultinomialNB()

# Train
model.fit(x_train, y_train)

# Save
pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model Saved Successfully")