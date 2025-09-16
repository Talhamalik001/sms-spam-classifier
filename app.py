import streamlit as st
import pickle
import os

st.title("📩 SMS Spam Classifier")

# Debug: check current directory and files
# st.write("Current directory:", os.getcwd())
# st.write("Files in directory:", os.listdir())

# Load model and vectorizer safely
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    st.success("Model and vectorizer loaded successfully!")
except Exception as e:
    st.error(f"Error loading model/vectorizer: {e}")

# Input
t_sms = st.text_area("Enter your message here:")

# Predict
if st.button("Predict"):
    if t_sms.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        try:
            vec = tfidf.transform([t_sms])
            result = model.predict(vec)[0]

            if result == 1:
                st.error("🚨 Spam Message")
            else:
                st.success("✅ Not Spam")
        except Exception as e:
            st.error(f"Error during prediction: {e}")




# import streamlit as st
# import pickle

# # Load model and vectorizer
# model = pickle.load(open("model.pkl", "rb"))
# tfidf = pickle.load(open("vectorizer.pkl", "rb"))

# # Streamlit UI
# st.title("📩 SMS Spam Classifier")

# t_sms = st.text_area("Enter your message here:")

# if st.button("Predict"):
#     if t_sms.strip() == "":
#         st.warning("⚠️ Please enter a message before predicting.")
#     else:
#         # Transform and predict
#         vec = tfidf.transform([t_sms])
#         result = model.predict(vec)[0]

#         if result == 1:
#             st.error("🚨 Spam Message")
#         else:
#             st.success("✅ Not Spam")



