import streamlit as st
import joblib
import io
import re

from docx import Document
import PyPDF2

# ---------------------------
# Text cleaning (same logic as training: clean_text_basic)
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
    # Keep this consistent with what you used to create df['clean_text']
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

def extract_text_from_upload(uploaded_file):
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
    else:
        # You can add .doc support via textract if needed
        raise ValueError("Unsupported file type. Please upload .pdf, .docx, or .txt")

# ---------------------------
# Load model
# ---------------------------

@st.cache_resource
def load_model():
    # best_model.pkl must be in the same folder as this app.py
    model = joblib.load("best_model.pkl")
    return model

model = load_model()

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("📄 Resume Classification App")
st.write(
    "Upload one or more resumes (.pdf / .docx / .txt) and the model will predict the job category."
)

uploaded_files = st.file_uploader(
    "Upload resume file(s)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"**Number of files uploaded:** {len(uploaded_files)}")

    if st.button("Predict Category"):
        import pandas as pd

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
        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True)

        # For convenience, show snippet of the first file
        first_ok = next((r for r in results if "ERROR" not in r["predicted_label"]), None)
        if first_ok:
            st.markdown("---")
            st.subheader("Sample Extracted Text (from first valid file)")
            # Need to re-read the first valid file for snippet
            first_file = next(f for f in uploaded_files if f.name == first_ok["filename"])
            text_snippet = extract_text_from_upload(first_file)[:1000]
            st.text(text_snippet)
else:
    st.info("Please upload at least one file to get predictions.")
