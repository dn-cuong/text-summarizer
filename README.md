# Text Summarizer

Abstractive summarization for English articles. Paste a long piece of prose and get a shorter rewrite - not a handful of sentences copied from the original.

The app is a FastAPI service around Hugging Face `sshleifer/distilbart-cnn-12-6`, with a simple reading UI at `/`.

## How it works

1. The request is tokenized with DistilBART’s tokenizer.
2. DistilBART only accepts **1024 tokens**. Longer articles are split into overlapping chunks (~900 tokens), summarized independently, then reduced into one summary.
3. Summary length is clamped so `max_length` never exceeds the input length (avoids generation errors on short text).
4. Each response includes character counts and inference latency in milliseconds.

The first request downloads the model weights (~300 MB). Later requests reuse the loaded pipeline.

English news and essays work best. The model was trained on CNN / DailyMail, so other languages will summarize poorly.

## Setup

Python 3.11+ recommended.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload
```

| | |
|---|---|
| UI | http://127.0.0.1:8000 |
| Health | http://127.0.0.1:8000/health |
| API docs | http://127.0.0.1:8000/docs |

## API

`POST /summarize`

```bash
curl -X POST http://127.0.0.1:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "<paste a long article here>",
    "max_length": 130,
    "min_length": 30
  }'
```

Request

```json
{
  "text": "string",
  "max_length": 130,
  "min_length": 30
}
```

`min_length` must be less than `max_length`. Both are token counts for the generated summary.

Response

```json
{
  "summary": "...",
  "input_length_chars": 973,
  "summary_length_chars": 286,
  "latency_ms": 2281.7
}
```

## Project layout

```
main.py              FastAPI app, chunking, summarization
static/index.html    Web UI
requirements.txt
```

## Model

Default: [`sshleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6) - a distilled BART, fast enough for a local demo.

For higher quality and more compute, change `MODEL_NAME` in `main.py` to `facebook/bart-large-cnn`.
