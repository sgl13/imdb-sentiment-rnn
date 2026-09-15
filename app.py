# Step 1: Import Libraries
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# Step 2: Load the IMDB word index
word_index = imdb.get_word_index()


# Step 3: Load the trained Simple RNN model
model = load_model("simple_rnn_imdb.keras")


# Step 4: Preprocess user input
def preprocess_text(text):

    # Convert text to lowercase and split into words
    words = text.lower().split()

    # Convert words to IMDB integer IDs
    encoded_review = [
        word_index.get(word, 2) + 3
        for word in words
    ]

    # Pad sequence to 100 words
    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=100,
        padding="pre"
    )

    return padded_review


# Step 5: Prediction function
def predict_sentiment(review):

    # Preprocess the review
    preprocessed_input = preprocess_text(review)

    # Make prediction
    prediction = model.predict(
        preprocessed_input,
        verbose=0
    )

    # Get prediction score
    score = float(prediction[0][0])

    # Classify sentiment
    if score >= 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return sentiment, score


# Step 6: Streamlit UI
st.title("IMDB Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review to classify it as positive or negative."
)


# Step 7: User input
user_input = st.text_area(
    "Movie Review",
    placeholder="Example: This movie was amazing and I really enjoyed it!"
)


# Step 8: Classification
if st.button("Classify"):

    if user_input.strip():

        sentiment, score = predict_sentiment(user_input)

        st.write(f"### Sentiment: {sentiment}")
        st.write(f"Prediction Score: {score:.4f}")

    else:

        st.warning("Please enter a movie review.")