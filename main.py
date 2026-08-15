"""
Text Summarizer API
--------------------
A small FastAPI service that summarizes text/article content using a
pretrained HuggingFace transformers summarization pipeline.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload

Then POST to /summarize:
    curl -X POST http://127.0.0.1:8000/summarize \
         -H "Content-Type: application/json" \
         -d '{"text": "<long article text here>", "max_length": 130, "min_length": 30}'
"""

import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import pipeline

app = FastAPI(title="Text Summarizer API", version="0.1.0")
STATIC_DIR = Path(__file__).parent / "static"
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
# DistilBART's position embeddings stop at 1024 tokens. Longer input
# raises IndexError unless we truncate or chunk first.
MAX_INPUT_TOKENS = 1024
CHUNK_TOKENS = 900
CHUNK_OVERLAP = 80


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The article/document text to summarize")
    max_length: int = Field(130, ge=10, le=1024, description="Max tokens in the summary")
    min_length: int = Field(30, ge=5, le=512, description="Min tokens in the summary")


class SummarizeResponse(BaseModel):
    summary: str
    input_length_chars: int
    summary_length_chars: int
    latency_ms: float


@lru_cache(maxsize=1)
def get_summarizer():
    # distilbart is smaller/faster than facebook/bart-large-cnn, good for a same-day demo.
    # Swap MODEL_NAME for facebook/bart-large-cnn if you want higher quality.
    return pipeline("summarization", model=MODEL_NAME, truncation=True)


def _token_ids(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def chunk_text(tokenizer, text: str) -> list[str]:
    ids = _token_ids(tokenizer, text)
    if len(ids) <= CHUNK_TOKENS:
        return [text]

    chunks = []
    start = 0
    while start < len(ids):
        end = min(start + CHUNK_TOKENS, len(ids))
        chunks.append(tokenizer.decode(ids[start:end], skip_special_tokens=True))
        if end == len(ids):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def run_summarizer(summarizer, text: str, min_length: int, max_length: int) -> str:
    tokenizer = summarizer.tokenizer
    input_len = len(tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=MAX_INPUT_TOKENS))
    # Generation max_length must stay below the input length for summarization.
    max_len = min(max_length, max(10, input_len - 2))
    min_len = min(min_length, max_len - 1)
    min_len = max(5, min_len)

    result = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        truncation=True,
    )
    if not result:
        raise RuntimeError("Model returned an empty summary")
    return result[0]["summary_text"].strip()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    if req.min_length >= req.max_length:
        raise HTTPException(status_code=400, detail="min_length must be less than max_length")

    summarizer = get_summarizer()
    started = time.perf_counter()
    try:
        chunks = chunk_text(summarizer.tokenizer, req.text)
        partials = [
            run_summarizer(summarizer, chunk, req.min_length, req.max_length)
            for chunk in chunks
        ]
        summary_text = partials[0] if len(partials) == 1 else run_summarizer(
            summarizer,
            " ".join(partials),
            req.min_length,
            req.max_length,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    return SummarizeResponse(
        summary=summary_text,
        input_length_chars=len(req.text),
        summary_length_chars=len(summary_text),
        latency_ms=latency_ms,
    )