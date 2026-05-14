---
title: SHL Assessment Recommender
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# SHL Assessment Recommender

A stateless, production-grade REST API that uses a Retrieval-Augmented Generation (RAG) pipeline to guide hiring professionals from a vague intent to a catalog-grounded shortlist of SHL Individual Test Solutions through natural dialogue.

## Architecture

- **FastAPI** — async REST API (`/health`, `/chat`)
- **Groq LLM** (`llama-3.3-70b-versatile`) — conversation agent
- **FAISS + sentence-transformers** — semantic catalog retrieval
- **RAG pipeline** — catalog-grounded recommendations, zero hallucination

## Quick Start

### 1. Install dependencies

```bash
cd shl-assessment-recommender
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your API key (already pre-filled for Groq):

```bash
cp .env.example .env
```

### 3. Build the FAISS index

This step scrapes the SHL catalog (with static fallback) and builds the vector index:

```bash
python scripts/build_index.py
```

### 4. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need to hire a mid-level Java developer. What assessments do you recommend?"}
    ]
  }'
```

## API Reference

### `GET /health`

Returns `{"status": "ok"}` when the service is running.

### `POST /chat`

**Request body:**
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Response body:**
```json
{
  "reply": "Based on your requirements...",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
      "test_type": ["Knowledge & Skills"]
    }
  ],
  "end_of_conversation": false
}
```

**Rules:**
- Max 8 user turns per conversation
- All recommendations are grounded in the SHL catalog — no hallucinations
- 30-second request timeout enforced

## Conversation Behavior

| State | Trigger | Action |
|---|---|---|
| **Clarify** | Turn 1 AND vague query | Ask one focused clarifying question |
| **Recommend** | Sufficient context (role + qualifier) | Return 3–10 catalog-grounded assessments |
| **Refine** | User edits constraints | Update the shortlist |
| **Compare** | User asks comparison | Compare specific assessments |
| **Refuse** | Prompt injection / off-topic | Polite refusal, no recommendations |

## Running Tests

```bash
pytest tests/ -v
```

## Running Evaluation

Start the API first, then:

```bash
python scripts/run_eval.py --base-url http://localhost:8000
```

Evaluation metrics computed:
- **Schema Compliance** — 100% target
- **Recall@10** — maximized
- **URL Integrity** — 100% target
- **Turn Cap Compliance** — 100% target
- **Hallucination Rate** — 0% target

## Docker

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 --env-file .env shl-recommender
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | LLM backend | `groq` |
| `LLM_API_KEY` | Groq API key | — |
| `LLM_MODEL` | Model name | `llama-3.3-70b-versatile` |
| `VECTOR_STORE_PATH` | FAISS index path | `data/faiss.index` |
| `CATALOG_PATH` | Catalog JSON path | `data/catalog.json` |
| `EMBED_MODEL` | Sentence Transformers model | `all-MiniLM-L6-v2` |
| `MAX_TURNS` | Max conversation turns | `8` |
| `REQUEST_TIMEOUT_S` | Per-request timeout | `30` |
| `TOP_K_RETRIEVAL` | Number of catalog entries retrieved | `15` |

## Project Structure

```
shl-assessment-recommender/
├── app/          # FastAPI service
├── agent/        # Conversation agent (state machine, guardrails, LLM)
├── retrieval/    # FAISS vector store + semantic retriever
├── ingestion/    # Catalog scraper, parser, indexer
├── data/         # catalog.json, faiss.index, catalog_metadata.json
├── evaluation/   # Harness, metrics, traces
├── scripts/      # build_index.py, run_eval.py
└── tests/        # pytest test suite
```
