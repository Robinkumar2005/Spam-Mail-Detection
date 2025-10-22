# import streamlit as st
# import joblib
# import re
# import numpy as np

# # ===== Load Models =====
# svm_model = joblib.load('svm_model.pkl')
# tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# # ===== Label Mapping =====
# label_map = {
#     0: ("Ham (Not Spam)", "✅", "linear-gradient(90deg, #2e7d32, #1b5e20)"),  # Dark green gradient
#     1: ("Spam", "🚫", "linear-gradient(90deg, #b71c1c, #7f0000)")           # Dark red gradient
# }

# # ===== Preprocess Text =====
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text)
#     text = re.sub(r'\W+', ' ', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# # ===== Highlight Spammy Words =====
# spammy_words = ["free", "win", "prize", "gift", "offer", "cash", "winner", "buy now", "urgent"]

# def highlight_spammy_words(text):
#     for word in spammy_words:
#         # Use regex for word boundaries to avoid partial matches
#         text = re.sub(f"\\b({re.escape(word)})\\b", r'<span style="background-color:yellow;color:black;font-weight:bold">\1</span>', text, flags=re.IGNORECASE)
#     return text

# # ===== Predict Function =====
# def predict_spam(text):
#     text_clean = preprocess_text(text)
#     X_vect = tfidf_vectorizer.transform([text_clean])
#     pred = svm_model.predict(X_vect)[0]
#     label, emoji, gradient = label_map[pred]
#     return label, emoji, gradient

# # ===== Streamlit UI =====
# st.set_page_config(page_title="📧 Email Spam Detection", layout="wide")
# st.title("📧 Email Spam Detection")
# st.write("Detect whether emails are **Spam** or **Ham**. You can paste multiple emails; each should start with 'Subject:'.")

# # ===== Sidebar =====
# st.sidebar.header("Tips")
# st.sidebar.write("- Start each email with `Subject:`")
# st.sidebar.write("- Provide complete sentences for better context")
# st.sidebar.write("- Avoid extremely short messages")

# # ===== User Input =====
# user_input = st.text_area("Paste your emails here:")

# if st.button("Predict Emails"):
#     if user_input.strip() == "":
#         st.warning("Please enter some email text to predict.")
#     else:
#         # Split emails whenever a new 'Subject:' appears
#         emails = [e.strip() for e in re.split(r'(?i)(?=Subject:)', user_input) if e.strip()]
#         st.write(f"Detected {len(emails)} email(s):")

#         for i, email in enumerate(emails, 1):
#             label, emoji, gradient = predict_spam(email)
#             email_highlighted = highlight_spammy_words(email)

#             # Expandable/collapsible section
#             with st.expander(f"Email {i} Prediction: {emoji} {label}"):
#                 st.markdown(
#                     f"<div style='background: {gradient}; padding:15px; border-radius:10px; color:white; font-weight:bold;'>"
#                     f"{email_highlighted}"
#                     f"</div>",
#                     unsafe_allow_html=True
#                 )

#                 # Copy prediction button
#                 st.code(f"Prediction: {label} {emoji}", language="text")
#                 st.button(f"Copy Prediction {i}", key=f"copy{i}")


import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd

# ===== 1️⃣ Load Models =====
svm_model = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# ===== 2️⃣ Label Mapping =====
label_map = {
    0: ("Ham (Not Spam)", "✅", "linear-gradient(90deg, #2e7d32, #1b5e20)"),  # Dark green gradient
    1: ("Spam", "🚫", "linear-gradient(90deg, #b71c1c, #7f0000)")           # Dark red gradient
}

# ===== 3️⃣ Preprocess Text =====
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ===== 4️⃣ Highlight Spammy Words with Tooltip =====
spammy_words = {
    "free": "Common spam trigger word",
    "win": "Common spam trigger word",
    "prize": "Common spam trigger word",
    "gift": "Common spam trigger word",
    "offer": "Common spam trigger word",
    "cash": "Common spam trigger word",
    "winner": "Common spam trigger word",
    "buy now": "Common spam trigger phrase",
    "urgent": "Common spam trigger word"
}

def highlight_spammy_words(text):
    for word, tooltip in spammy_words.items():
        text = re.sub(f"\\b({re.escape(word)})\\b",
                      f'<span style="background-color:yellow;color:black;font-weight:bold" title="{tooltip}">\\1</span>',
                      text, flags=re.IGNORECASE)
    return text

# ===== 5️⃣ Predict Function (Batch) =====
def predict_spam_batch(emails):
    emails_clean = [preprocess_text(email) for email in emails]
    X_vect = tfidf_vectorizer.transform(emails_clean)
    preds = svm_model.predict(X_vect)
    results = [label_map[pred] for pred in preds]
    return results

# ===== 6️⃣ Streamlit UI =====
st.set_page_config(page_title="📧 Email Spam Detector", layout="wide")
st.title("📧 Modern Email Spam Detector")
st.write("""
Detect whether emails are **Spam** or **Ham**.  
You can paste **multiple emails**, each starting with 'Subject:'.  
Interactive features: highlighted spammy words, scrollable email previews, and batch predictions.
""")

# ===== Sidebar =====
st.sidebar.header("Tips & Tricks")
st.sidebar.write("""
- Start each email with `Subject:`  
- Provide complete sentences for better context  
- Avoid extremely short messages  
- Common spam words: free, win, prize, gift, offer, cash
""")

# ===== User Input =====
user_input = st.text_area("Paste your emails here:")

if st.button("Predict Emails"):
    if user_input.strip() == "":
        st.warning("Please enter some email text to predict.")
    else:
        # Split multiple emails
        emails = [e.strip() for e in re.split(r'(?i)(?=Subject:)', user_input) if e.strip()]
        st.write(f"Detected {len(emails)} email(s):")

        results = predict_spam_batch(emails)
        all_predictions = []

        for i, (email, (label, emoji, gradient)) in enumerate(zip(emails, results), 1):
            email_highlighted = highlight_spammy_words(email)
            all_predictions.append({"Email": f"Email {i}", "Prediction": label, "Emoji": emoji, "Content": email})

            # ===== Expandable Scrollable Email Card =====
            with st.expander(f"Email {i} Prediction: {emoji} {label}"):
                st.markdown(
                    f"""
                    <div style='background: {gradient};
                                padding:15px;
                                border-radius:10px;
                                color:white;
                                font-weight:bold;
                                max-height:250px;
                                overflow-y:auto;
                                box-shadow: 0px 4px 10px rgba(0,0,0,0.3);'>
                        {email_highlighted}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.code(f"Prediction: {label} {emoji}", language="text")

        # ===== Download CSV Button =====
        df = pd.DataFrame(all_predictions)
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name='email_predictions.csv',
            mime='text/csv'
        )
