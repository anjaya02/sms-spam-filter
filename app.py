import streamlit as st
import pickle
import string

from nltk.tokenize       import wordpunct_tokenize
from nltk.corpus         import stopwords
from nltk.stem.porter    import PorterStemmer

# 1. Load the trained artifacts
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl",     "rb"))

# 2. Pre-setup
ps         = PorterStemmer()
stop_words = set(stopwords.words("english"))

def transform_text(text):
    tokens = wordpunct_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    stems  = [ps.stem(t) for t in tokens]
    return " ".join(stems)

# 3. Streamlit UI
st.title("📨 Email/SMS Spam Classifier")
input_sms = st.text_area("Enter your message:")

if st.button("Predict"):
    cleaned = transform_text(input_sms)
    vec     = tfidf.transform([cleaned])
    pred    = model.predict(vec)[0]

    if pred == 1:
        st.error("🚨 Spam")
    else:
        st.success("✅ Not Spam")
