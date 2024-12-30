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

st.text("sentiment analysis")
text = st.text_input("please enter your review")
state = st.button("predict")

token = helper.preprocessing_step(text)
vectorized_data = vectorizer.transform([token])
prediction = model.predict(vectorized_data)

if state :
    st.text(prediction)
