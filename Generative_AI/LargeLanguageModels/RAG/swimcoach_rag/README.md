# SwimCoach RAG System
### Azure OpenAI + Pinecone + Azure Cognitive Search  
**Assignment 1 – Generative AI | Professional Swimming Coach RAG Solution**

---

## Architecture Overview

```
VIDEO → ffmpeg chunks → OpenCV frames → GPT-4o Vision captions
      → GPT-4o summarise → Ada-002 embed → Pinecone + ACS + Cosmos DB

QUERY → GPT-4o decompose → Parallel (BM25 + Semantic) → RRF Fusion
      → GPT-4o Reranker → GPT-4 Turbo SwimCoachGPT → Streamed response
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.11 |
| ffmpeg | system install (`brew install ffmpeg` / `apt install ffmpeg`) |
| Azure OpenAI | Deployments: `gpt-4o`, `gpt-4-turbo`, `text-embedding-ada-002` |
| Pinecone | Serverless account (free tier works for ≤850 vectors) |
| Azure Cognitive Search | Standard S1 (for BM25) |
| Azure Cosmos DB | Core SQL API (optional – for full metadata persistence) |

---

## Setup

### 1. Clone & install

```bash
cd swimcoach_rag
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in all values in .env
```

Required keys in `.env`:
```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_GPT4O_DEPLOYMENT=gpt-4o
AZURE_OPENAI_GPT4_TURBO_DEPLOYMENT=gpt-4-turbo
AZURE_OPENAI_ADA_DEPLOYMENT=text-embedding-ada-002

PINECONE_API_KEY=...
PINECONE_INDEX_NAME=swimcoach-kb

AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_API_KEY=...
AZURE_SEARCH_INDEX_NAME=swimcoach-bm25

COSMOS_ENDPOINT=https://your-cosmos.documents.azure.com:443/  # optional
COSMOS_KEY=...                                                  # optional
```

---

## Usage

### Step 1 – Seed Knowledge Base (no video needed)

Loads all 8 sample videos + 5 coaching logs + 1 athlete profile into  
Pinecone, Azure Cognitive Search, and Cosmos DB:

```bash
python main.py seed
```

### Step 2 – Run the Primary Sample Query (Assignment §3.1)

```bash
python main.py query
```

Expected output (5-section SwimCoachGPT response):
```
## 1. Must-Watch Videos        → V-0091 flagged (92% completion, 1.3% improvement)
## 2. Videos to Avoid Now      → too-advanced content not yet appropriate
## 3. Gap Analysis             → videos watched by similar athletes but not this one
## 4. Suggested Viewing Sequence → ordered playlist
## 5. Confidence Level         → HIGH / MEDIUM / LOW
```

### Step 3 – Run a Custom Query

```bash
python main.py query --q "Which dolphin kick videos produced measurable improvement for male butterfly swimmers aged 18-22?"
```

### Step 4 – Ingest a New Coaching Video

```bash
python main.py ingest \
  --video /path/to/coaching_video.mp4 \
  --video-id V-0200 \
  --tags butterfly "dolphin kick" breakout underwater \
  --difficulty advanced
```

Pipeline stages:
1. `ffmpeg` splits video into ≤20 MB chunks
2. `OpenCV` samples 1 frame per 2.5 s per chunk
3. `GPT-4o Vision` captions each frame (1-3 sentences)
4. `GPT-4o` synthesises all captions → 10-200 word `embedding_text`
5. `Ada-002` embeds the summary → 1536-dim vector
6. Upserted to **Pinecone** + **Azure Cognitive Search** + **Cosmos DB**

### Step 5 – Interactive REPL

```bash
python main.py repl
```

### Step 6 – Run All Sample Queries (Demo Mode)

```bash
python main.py demo
```

---

## Project Structure

```
swimcoach_rag/
├── main.py                   ← CLI entry point (seed/ingest/query/repl/demo)
├── requirements.txt
├── .env.example
│
├── config/
│   └── settings.py           ← Azure OpenAI, Pinecone, ACS, Cosmos clients
│
├── utils/
│   └── helpers.py            ← embed_text, caption_frame, summarise_captions
│
├── kb/
│   └── seed_data.py          ← 8 video catalog entries + 5 coaching logs + 1 profile
│
├── ingestion/
│   └── pipeline.py           ← 5-stage video → vector store pipeline
│
└── query/
    └── pipeline.py           ← 5-stage RAG query pipeline (streamed)
```

---

## Pipeline Stage Details

### Data Ingestion

| Stage | Component | Detail |
|-------|-----------|--------|
| 1 | ffmpeg Chunker | Splits video into ≤20 MB segments (`-f segment -c copy`) |
| 2 | OpenCV Frame Extractor | 1 frame per 2.5 s, saved as JPEG |
| 3 | GPT-4o Vision Caption | 1-3 sentences per frame (stroke phase, body position, kick mechanics) |
| 4 | GPT-4o Summariser | All captions → 10-200 word `embedding_text` |
| 5 | Ada-002 Embedder | 1536-dim vector → Pinecone + ACS BM25 + Cosmos DB |

### RAG Query

| Stage | Component | Detail |
|-------|-----------|--------|
| 1 | Query Decomposer | GPT-4o extracts `{event, weakness_tags, age_range, gender, weeks_to_competition}` |
| 2 | Parallel Retrieval | BM25 (ACS top-10) + Semantic (Pinecone ANN top-10), run concurrently |
| 3 | RRF Fusion | `0.5×1/(60+rank_BM25) + 0.5×1/(60+rank_Semantic)` → top-20 |
| 4 | LLM Reranker | GPT-4o scores each of top-10 candidates 1-10 → top-5 |
| 5 | SwimCoachGPT | GPT-4 Turbo generates 5-section streamed coaching recommendation |

---

## Cost Estimate (at 1M users, per assignment §7.1)

| Service | Monthly |
|---------|---------|
| Azure OpenAI (GPT-4o + Turbo + Ada) | ~$8,000 |
| Pinecone Serverless | ~$1,200 |
| AKS cluster | ~$1,500 |
| ACS + Cosmos + Redis | ~$750 |
| **Total** | **~$11,450/month** |

Redis cache (40% hit rate) saves ~40% of LLM costs.  
Reserved instances + spot nodes reduce AKS cost by ~30%.

---

## Success Metrics (§7.2)

| Metric | Target |
|--------|--------|
| Precision@5 | > 0.70 |
| Recall@5 | > 0.65 |
| MRR | > 0.75 |
| Coach Adoption Rate | > 60% |
| P95 Query Latency | < 2 s |
| Cost per active user | < $0.012/month |
