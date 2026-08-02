"""
Streamlit demo UI.
Talks to the FastAPI backend -- run backend first:
    uvicorn app.main:app --reload
Then run this:
    streamlit run frontend/streamlit_app.py
"""
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Document QA", layout="centered")
st.title("📄 RAG Document QA")

st.header("1. Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
if uploaded_file and st.button("Ingest Document"):
    with st.spinner("Extracting, chunking, embedding..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        resp = requests.post(f"{API_URL}/upload", files=files)
        if resp.ok:
            data = resp.json()
            st.success(f"Indexed {data['chunks_indexed']} chunks from {data['filename']}")
        else:
            st.error(f"Upload failed: {resp.text}")

st.header("2. Ask a question")
question = st.text_input("Your question")
if question and st.button("Ask"):
    with st.spinner("Retrieving and generating..."):
        resp = requests.post(f"{API_URL}/ask", data={"question": question})
        if resp.ok:
            data = resp.json()
            st.subheader("Answer")
            st.write(data["answer"])

            st.subheader("Sources")
            for c in data["citations"]:
                marker = "✅" if c["source_number"] in data["cited_source_numbers"] else "—"
                st.markdown(
                    f"{marker} **[Source {c['source_number']}]** "
                    f"{c['filename']}, page {c['page_number']}"
                )
                st.caption(c["snippet"])
        else:
            st.error(f"Request failed: {resp.text}")
