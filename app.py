import streamlit as st
import joblib
import io
import re

from docx import Document
import PyPDF2

import nltk
from nltk.corpus import stopwords

# Optional: support for .doc using textract
try:
    import textract
except ImportError:
    textract = None

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

def read_doc_file(file_bytes):
    """
    Optional reader for .doc using textract.
    Make sure 'textract' is in requirements.txt if you use this.
    """
    if textract is None:
        # Graceful error if textract is not installed
        raise ValueError("DOC support requires 'textract'. Please install textract or convert DOC to DOCX.")
    text = textract.process(io.BytesIO(file_bytes))
    return text.decode("utf-8", errors="ignore")

def extract_text_from_upload(uploaded_file):
    """
    Extract text from a single uploaded file object.
    """
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
    elif name.endswith(".doc"):
        return read_doc_file(file_bytes)
    else:
        raise ValueError("Unsupported file type. Please upload .pdf, .docx, .doc, or .txt")

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
    "Upload one or more resumes (.pdf / .docx / .doc / .txt) and the model will predict the job category."
)

uploaded_files = st.file_uploader(
    "Upload your resume file(s)",
    type=["pdf", "docx", "doc", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"**Number of files uploaded:** {len(uploaded_files)}")

    if st.button("Predict Category for All Files"):
        results = []
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            try:
                # Important: read per file inside loop
                raw_text = extract_text_from_upload(uploaded_file)

                if not raw_text.strip():
                    results.append({
                        "filename": file_name,
                        "predicted_label": "ERROR: No text extracted",
                        "details": "Empty or unreadable file"
                    })
                    continue

                processed_text = preprocess_text(raw_text)
                pred_label = model.predict([processed_text])[0]

                # Default: no probability
                prob_str = "N/A"

                # Show probabilities if available
                clf = model.named_steps.get('clf', None)
                if clf is not None and hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(
                        model.named_steps['tfidf'].transform([processed_text])
                    )[0]
                    classes = clf.classes_
                    # pick max confidence and label
                    max_idx = probs.argmax()
                    prob_str = f"{classes[max_idx]}: {probs[max_idx]:.3f}"

                results.append({
                    "filename": file_name,
                    "predicted_label": str(pred_label),
                    "top_confidence": prob_str
                })

            except Exception as e:
                results.append({
                    "filename": file_name,
                    "predicted_label": "ERROR",
                    "top_confidence": str(e)
                })

        st.subheader("Prediction Results")
        import pandas as pd
        res_df = pd.DataFrame(results)
        st.dataframe(res_df)

        # Optionally show details for first file
        st.write("You can scroll the table above to see all predictions.")
