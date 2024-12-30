import streamlit as st
import sklearn
import helper
import pickle
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
model=pickle.load(open("Models/model.pkl",'rb'))
vectorizer=pickle.load(open("Models/vectorizer.pkl",'rb'))

# Load the model and vectorizer
try:
    model = pickle.load(open("Models/model.pkl", 'rb'))
    vectorizer = pickle.load(open("Models/vectorizer.pkl", 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer files not found. Please check the file paths.")
    st.stop()

# Title of the app
st.title("Sentiment Analysis App using ML")

# User input
text = st.text_input("Enter your review")
state = st.button("Predict")

# Sentiment mapping dictionary
sentiment_dict = {
    0: "Negative",
    1: "Positive"
}

# Add a custom CSS for balloons
st.markdown(
    """
    <style>
    .balloon {
        padding: 20px;
        margin-top: 20px;
        border-radius: 15px;
        background-color: #f0f8ff; /* Light blue */
        color: #000; /* Black text */
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
    }
    .positive {
        background-color: #d4edda; /* Light green */
        color: #155724; /* Dark green text */
    }
    .negative {
        background-color: #f8d7da; /* Light red */
        color: #721c24; /* Dark red text */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Perform prediction
if state:
    if text.strip() == "":
        st.error("Please enter a valid review text.")
    else:
        try:
            # Preprocess the input text
            token = helper.preprocessing_step(text)

            # Vectorize the preprocessed text
            vectorized_data = vectorizer.transform([token])

            # Predict the sentiment
            prediction = model.predict(vectorized_data)

            # Map prediction to sentiment
            sentiment = sentiment_dict[prediction[0]]  # Use prediction[0] to get the value

            # Set the balloon style based on sentiment
            if sentiment == "Positive":
                balloon_class = "balloon positive"
                st.balloons()
            else:
                Snow_class = "Snow negative"
                st.snow()

            # Display the sentiment in a styled balloon
            st.markdown(f'<div class="{balloon_class}">The sentiment is: {sentiment}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
