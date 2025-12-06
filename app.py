import streamlit as st
import joblib
import io
import re

from docx import Document
import PyPDF2

import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords are available
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# ---------------------------
# Text cleaning (same as training)
# ---------------------------

def clean_text_basic(s):
    s = str(s).lower()
    s = re.sub(r'\s+', ' ', s)            # collapse whitespace
    s = re.sub(r'http\S+', '', s)         # remove urls
    s = re.sub(r'\@\w+', '', s)           # remove @mentions
    s = re.sub(r'[^a-z0-9\s]', ' ', s)    # keep alphanumeric + space
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def remove_stopwords(s):
    tokens = s.split()
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

def preprocess_text(s):
    s = clean_text_basic(s)
    s = remove_stopwords(s)
    return s

# ---------------------------
# File readers for uploaded resumes
# ---------------------------

def read_docx_file(file_bytes):
    file_stream = io.BytesIO(file_bytes)
    doc = Document(file_stream)
    return "\n".join(p.text for p in doc.paragraphs)

def read_pdf_file(file_bytes):
    file_stream = io.BytesIO(file_bytes)
    reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def read_txt_file(file_bytes):
    return file_bytes.decode("utf-8", errors="ignore")

def extract_text_from_upload(uploaded_file):
    if uploaded_file is None:
        return ""

    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".docx"):
        return read_docx_file(file_bytes)
    elif name.endswith(".pdf"):
        return read_pdf_file(file_bytes)
    elif name.endswith(".txt"):
        return read_txt_file(file_bytes)
    else:
        raise ValueError("Unsupported file type. Please upload .pdf, .docx, or .txt")

# ---------------------------
# Load model
# ---------------------------

@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    return model

model = load_model()

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("📄 Resume Classification App")
st.write(
    "Upload a resume (.pdf / .docx / .txt) and the model will predict the job category."
)

uploaded_file = st.file_uploader(
    "Upload your resume file",
    type=["pdf", "docx", "txt"]
)

if uploaded_file is not None:
    st.write("**File uploaded:**", uploaded_file.name)

    if st.button("Predict Category"):
        try:
            raw_text = extract_text_from_upload(uploaded_file)

            if not raw_text.strip():
                st.error("Could not extract any text from the file.")
            else:
                st.subheader("Extracted Text (first 1000 chars)")
                st.text(raw_text[:1000])

                processed_text = preprocess_text(raw_text)
                pred_label = model.predict([processed_text])[0]
                st.subheader("Predicted Category")
                st.success(pred_label)

                # Show probabilities if available
                clf = model.named_steps.get('clf', None)
                if clf is not None and hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(
                        model.named_steps['tfidf'].transform([processed_text])
                    )[0]
                    classes = clf.classes_

                    st.subheader("Prediction Probabilities")
                    prob_table = {cls: float(p) for cls, p in zip(classes, probs)}
                    st.write(prob_table)

        except Exception as e:
            st.error(f"Error processing file: {e}")
