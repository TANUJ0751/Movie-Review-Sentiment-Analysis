import streamlit as st
from joblib import load
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

model=load('Random_Forest.joblib')

vectorizer=load('vectorizer.joblib')

#Streamlit app
st.title("Movie Review Sentiment Analysis App")
st.write("This app uses a Random Forest Classifier to predict the sentiment of a movie review.")
st.write("Created and Developed by Tanuj jain")

#input Field
user_input=st.text_area("Enter Your Review : ","")

if st.button("Analyze Sentiment"):
    if user_input.strip()=="":
        st.write("Enter a Valid Review ")
    else:
        #Preprocess input
        stopwords=set(stopwords.words('english'))
        preprocessed_text=" ".join([word for word in user_input.split() if word not in stopwords])

        #convert input to feature vector
        input_vector=vectorizer.transform([preprocessed_text])

        #predict Sentiment
        prediction=model.predict(input_vector)[0]
        sentiment="Positive" if prediction ==1 else "Negative"

        #Display the result
        st.subheader(f"Sentiment : {sentiment}")
        

