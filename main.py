import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
ps=PorterStemmer()
def text_transformation(text):
    y = []
    text = text.lower()
    text = nltk.word_tokenize(text)

    for i in text:
        if i.isalnum():
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)
    return " ".join(y)


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
st.title("Email/Spam Classifier")
input_sms=st.text_area("Enter the message")

if st.button("Predict"):
    transformed_sms=text_transformation(input_sms)
    vector_input=tfidf.transform([transformed_sms])
    result=model.predict(vector_input)[0]

    if result==1:
        st.header('spam')
    else:
        st.header("Not Spam")
