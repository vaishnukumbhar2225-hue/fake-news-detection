import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("Fake News Detection App")

# Input box
news = st.text_area("Enter News Text")

# Button
if st.button("Check News"):

    # Transform text
    news_vector = vectorizer.transform([news])

    # Prediction
    prediction = model.predict(news_vector)

    # Result
    if prediction[0] == 0:
        st.error("Fake News")
    else:
        st.success("Real News")