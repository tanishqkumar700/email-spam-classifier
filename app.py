import streamlit as st
import pickle

# Set page configuration
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="centered")

# 1. Load the saved model and vectorizer safely
@st.cache_resource  # Keeps the model cached in memory so it doesn't reload on every click
def load_assets():
    with open('spam_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_assets()
except FileNotFoundError:
    st.error("⚠️ Error: 'spam_model.pkl' or 'vectorizer.pkl' not found in the current directory.")
    st.stop()

# 2. UI Layout Design
st.title("📧 Email Spam Classifier")
st.write("An intelligent application powered by a **Naive Bayes Classifier** to instantly detect spam messages.")
st.markdown("---")

# User Text Input
user_input = st.text_area(
    "Paste the email content below:", 
    height=200, 
    placeholder="Type or paste text here (e.g., 'Congratulations! You've won a $1,000 Walmart gift card...')"
)

# 3. Prediction & Execution Logic
if st.button("Analyze Email", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter some text before classifying.")
    else:
        with st.spinner("Processing text and running classification..."):
            # Transform text using the exact same vectorizer from Kaggle
            transformed_text = vectorizer.transform([user_input])
            
            # Predict
            prediction = model.predict(transformed_text)[0]
            
            # Display localized styling based on binary classification results
            # Handles both integer (0/1) and string ('spam'/'ham') label outputs
            st.markdown("### **Classification Result:**")
            if prediction == 1 or str(prediction).lower() == 'spam':
                st.error("🚨 **SPAM DETECTED**")
                st.info("This email shows high algorithmic probability of being a phishing attempt or unsolicited spam.")
            else:
                st.success("✅ **HAM (LEGITIMATE EMAIL)**")
                st.info("This message looks clean and safe to read.")