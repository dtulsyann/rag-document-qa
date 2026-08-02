"""
FastAPI wrapper.

Thin HTTP layer over pipeline.py -- no retrieval logic lives here.
Run with: uvicorn app.main:app --reload
"""
import os
import shutil

from fastapi import FastAPI, UploadFile, File, Form
from app.pipeline import ingest_pdf, answer_question
from app.config import RAW_DIR, PipelineConfig

app = FastAPI(title="RAG Document QA")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs(RAW_DIR, exist_ok=True)
    save_path = os.path.join(RAW_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    config = PipelineConfig(collection_name="baseline")
    num_chunks = ingest_pdf(save_path, config=config)

    return {"filename": file.filename, "chunks_indexed": num_chunks}


@app.post("/ask")
async def ask(question: str = Form(...)):
    config = PipelineConfig(collection_name="baseline")
    result = answer_question(question, config=config)

    return {
        "answer": result.answer,
        "citations": result.citations,
        "cited_source_numbers": result.cited_source_numbers,
    }
