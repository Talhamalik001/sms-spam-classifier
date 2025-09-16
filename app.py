import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit UI
st.title("📩 SMS Spam Classifier")

t_sms = st.text_area("Enter your message here:")

if st.button("Predict"):
    if t_sms.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        # Transform and predict
        vec = tfidf.transform([t_sms])
        result = model.predict(vec)[0]

        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")

