import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("Fake News Detection App")

# User Input
news = st.text_area("Enter News Text")

# Prediction
if st.button("Check News"):
    transformed_news = vectorizer.transform([news])
    prediction = model.predict(transformed_news)

    if prediction[0] == 0:
        st.error("Fake News")
    else:
        st.success("Real News")