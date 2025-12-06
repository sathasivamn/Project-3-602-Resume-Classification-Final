import streamlit as st
import joblib
import io
import re

from docx import Document
import PyPDF2

# Optional: support for .doc (only if textract is installed)
try:
    import textract
except ImportError:
    textract = None

# ---------------------------
# Text cleaning (same as training)
# ---------------------------

def clean_text_basic(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r'\s+', ' ', s)            # collapse whitespace
    s = re.sub(r'http\S+', '', s)         # remove urls
    s = re.sub(r'\@\w+', '', s)           # remove @mentions
    s = re.sub(r'[^a-z0-9\s]', ' ', s)    # keep alphanumeric + space
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def preprocess_text(s: str) -> str:
    # Same preprocessing as df['clean_text'] in training
    return clean_text_basic(s)

# ---------------------------
# File readers for uploaded resumes
# ---------------------------

def read_docx_file(file_bytes: bytes) -> str:
    file_stream = io.BytesIO(file_bytes)
    doc = Document(file_stream)
    return "\n".join(p.text for p in doc.paragraphs)

def read_pdf_file(file_bytes: bytes) -> str:
    file_stream = io.BytesIO(file_bytes)
    reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def read_txt_file(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def read_doc_file(file_bytes: bytes) -> str:
    if textract is None:
        raise ValueError("DOC support requires 'textract'. Please install textract or convert DOC to DOCX.")
    text = textract.process(io.BytesIO(file_bytes))
    return text.decode("utf-8", errors="ignore")

def extract_text_from_upload(uploaded_file) -> str:
    """Extract text from a single uploaded file object."""
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
    "Upload resume file(s)",
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
                raw_text = extract_text_from_upload(uploaded_file)

                if not raw_text.strip():
                    results.append({
                        "filename": file_name,
                        "predicted_label": "ERROR: No text extracted",
                        "top_confidence": ""
                    })
                    continue

                processed_text = preprocess_text(raw_text)
                pred_label = model.predict([processed_text])[0]

                # Default: no probability
                prob_str = ""

                # Show probabilities if available
                clf = model.named_steps.get("clf", None)
                if clf is not None and hasattr(clf, "predict_proba"):
                    tfidf = model.named_steps["tfidf"]
                    probs = clf.predict_proba(tfidf.transform([processed_text]))[0]
                    classes = clf.classes_
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

# Build a simple HTML table instead of st.dataframe (to avoid pandas)
table_html = """
<table style="border-collapse: collapse; width: 100%;">
    <tr>
        <th style="border: 1px solid #ccc; padding: 4px;">Filename</th>
        <th style="border: 1px solid #ccc; padding: 4px;">Predicted Label</th>
        <th style="border: 1px solid #ccc; padding: 4px;">Top Confidence</th>
    </tr>
"""

for r in results:
    table_html += f"""
    <tr>
        <td style="border: 1px solid #ccc; padding: 4px;">{r['filename']}</td>
        <td style="border: 1px solid #ccc; padding: 4px;">{r['predicted_label']}</td>
        <td style="border: 1px solid #ccc; padding: 4px;">{r['top_confidence']}</td>
    </tr>
    """

table_html += "</table>"

st.markdown(table_html, unsafe_allow_html=True)
