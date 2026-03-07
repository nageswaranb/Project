"""
query/pipeline.py
─────────────────────────────────────────────────────────────────
RAG QUERY PIPELINE
  Coach Query
  → Stage 1: Query Decomposer       (GPT-4o structured filters)
  → Stage 2: Parallel Hybrid Search (BM25 via ACS + Semantic via Pinecone)
  → Stage 3: RRF Fusion             (50/50 Reciprocal Rank Fusion, k=60)
  → Stage 4: LLM Reranker           (GPT-4o scores top-10 → top-5)
  → Stage 5: SwimCoachGPT           (GPT-4 Turbo 5-section response)
  → Streamed response to coach

Run:
    python -m query.pipeline --query "Am I missing any must-watch videos for
    this athlete: 19M, 200m Butterfly, dolphin kick weakness, 6 weeks to State."
─────────────────────────────────────────────────────────────────
"""

import sys
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import (
    openai_client,
    GPT4O_DEPLOYMENT,
    GPT4_TURBO_DEPLOYMENT,
    get_pinecone_index,
    get_search_client,
)
from utils.helpers import embed_text

console = Console()

# ── In-domain guard phrase ─────────────────────────────────────────────────────
OUT_OF_DOMAIN_REPLY = (
    "Sorry, I'm unable to give the information. "
    "Please re-phrase your query with respect to my domain knowledge."
)

SWIM_KEYWORDS = {
    "swim", "butterfly", "freestyle", "backstroke", "breaststroke", "medley",
    "dolphin", "stroke", "kick", "athlete", "coach", "video", "drill",
    "technique", "race", "competition", "championship", "pool", "turn",
    "breakout", "split", "pacing", "streamline", "catch", "pull", "recovery",
}

def is_in_domain(query: str) -> bool:
    return any(kw in query.lower() for kw in SWIM_KEYWORDS)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 – Query Decomposer
# ══════════════════════════════════════════════════════════════════════════════

DECOMPOSE_SYSTEM = """You are a swimming coaching assistant query parser.
Extract structured filters from the coach's natural language query.
Return ONLY valid JSON with these fields (use null if unknown):
{
  "event": "e.g. 200m Butterfly",
  "weakness_tags": ["dolphin kick", "hip rotation"],
  "age_range": [18, 22],
  "gender": "M" | "F" | null,
  "weeks_to_competition": 6,
  "videos_already_watched": ["V-0103"],
  "intent": "gap_analysis" | "sequence" | "improvement" | "general",
  "difficulty_filter": "beginner" | "intermediate" | "advanced" | null
}"""

def decompose_query(query: str) -> dict:
    """Stage 1: Extract structured filters from free-text coach query."""
    console.log("[bold cyan]Stage 1:[/] Decomposing query …")
    response = openai_client.chat.completions.create(
        model=GPT4O_DEPLOYMENT,
        messages=[
            {"role": "system", "content": DECOMPOSE_SYSTEM},
            {"role": "user",   "content": query},
        ],
        response_format={"type": "json_object"},
        max_tokens=400,
        temperature=0,
    )
    filters = json.loads(response.choices[0].message.content)
    console.log(f"  Filters: {json.dumps(filters, indent=2)}")
    return filters


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 – Parallel Hybrid Retrieval
# ══════════════════════════════════════════════════════════════════════════════

def semantic_search(query: str, filters: dict, top_k: int = 10) -> list[dict]:
    """
    Pinecone ANN semantic search.
    Applies metadata pre-filters from decomposed query where possible.
    """
    query_vector = embed_text(query)
    pc_index = get_pinecone_index()

    # Build Pinecone metadata filter
    pc_filter = {}
    if filters.get("difficulty_filter"):
        pc_filter["difficulty"] = {"$eq": filters["difficulty_filter"]}

    results = pc_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=pc_filter if pc_filter else None,
    )

    hits = []
    for i, match in enumerate(results.matches):
        hits.append({
            "id":             match.id,
            "rank_semantic":  i + 1,
            "score_semantic": match.score,
            "metadata":       match.metadata,
            "source":         "semantic",
        })
    return hits


def bm25_search(query: str, filters: dict, top_k: int = 10) -> list[dict]:
    """
    Azure Cognitive Search BM25 keyword search.
    """
    search_client = get_search_client()

    # Optional OData filter
    odata_filter = None
    if filters.get("difficulty_filter"):
        odata_filter = f"difficulty eq '{filters['difficulty_filter']}'"

    results = search_client.search(
        search_text=query,
        top=top_k,
        filter=odata_filter,
        select=["id", "video_id", "doc_type", "difficulty", "embedding_text", "topic_tags_str"],
    )

    hits = []
    for i, r in enumerate(results):
        hits.append({
            "id":          r.get("video_id") or r["id"],
            "rank_bm25":   i + 1,
            "score_bm25":  r.get("@search.score", 0),
            "metadata": {
                "embedding_text": r.get("embedding_text", ""),
                "doc_type":       r.get("doc_type", ""),
                "difficulty":     r.get("difficulty", ""),
                "topic_tags":     r.get("topic_tags_str", "").split(),
            },
            "source": "bm25",
        })
    return hits


def parallel_retrieve(query: str, filters: dict) -> tuple[list, list]:
    """Run BM25 and Semantic search concurrently."""
    console.log("[bold cyan]Stage 2:[/] Parallel hybrid retrieval …")
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_sem = executor.submit(semantic_search, query, filters, 10)
        fut_bm25 = executor.submit(bm25_search,    query, filters, 10)
        sem_hits  = fut_sem.result()
        bm25_hits = fut_bm25.result()
    console.log(f"  Semantic: {len(sem_hits)} hits | BM25: {len(bm25_hits)} hits")
    return sem_hits, bm25_hits


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 – RRF Fusion (50/50, k=60)
# ══════════════════════════════════════════════════════════════════════════════

RRF_K = 60

def rrf_fusion(sem_hits: list[dict], bm25_hits: list[dict], top_n: int = 20) -> list[dict]:
    """
    Reciprocal Rank Fusion:
        RRF_score(d) = 0.5 × 1/(60 + rank_BM25) + 0.5 × 1/(60 + rank_Semantic)
    Returns top_n merged documents sorted by descending RRF score.
    """
    console.log("[bold cyan]Stage 3:[/] RRF fusion …")
    scores: dict[str, dict] = {}

    for hit in sem_hits:
        did = hit["id"]
        scores.setdefault(did, {"rrf": 0.0, "metadata": hit["metadata"]})
        scores[did]["rrf"] += 0.5 * (1 / (RRF_K + hit["rank_semantic"]))

    for hit in bm25_hits:
        did = hit["id"]
        scores.setdefault(did, {"rrf": 0.0, "metadata": hit["metadata"]})
        scores[did]["rrf"] += 0.5 * (1 / (RRF_K + hit["rank_bm25"]))

    merged = [{"id": k, **v} for k, v in scores.items()]
    merged.sort(key=lambda x: x["rrf"], reverse=True)
    top = merged[:top_n]
    console.log(f"  Merged {len(scores)} unique docs → top {len(top)} after RRF")
    return top


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 – LLM Reranker (GPT-4o)
# ══════════════════════════════════════════════════════════════════════════════

RERANKER_SYSTEM = """You are an expert swimming coach assistant acting as a reranker.
Given a coaching query and a set of candidate documents, score each document 1-10
based on its relevance to the query. Consider:
  - Athlete event match (butterfly vs freestyle etc.)
  - Weakness relevance (dolphin kick, catch mechanics etc.)
  - Age/experience appropriateness
  - Prerequisite ordering (beginner before advanced)
  - Recency and watch-history gaps
Return ONLY valid JSON array:
[{"doc_id": "...", "relevance_score": 8, "reason": "..."}]"""

def llm_rerank(query: str, candidates: list[dict], filters: dict, top_n: int = 5) -> list[dict]:
    """Stage 4: GPT-4o scores top-10 RRF candidates contextually."""
    console.log("[bold cyan]Stage 4:[/] LLM Reranking …")
    top10 = candidates[:10]

    candidate_text = "\n\n".join(
        f"doc_id: {c['id']}\n"
        f"type: {c['metadata'].get('doc_type','')}\n"
        f"difficulty: {c['metadata'].get('difficulty','')}\n"
        f"tags: {', '.join(c['metadata'].get('topic_tags', []))}\n"
        f"summary: {c['metadata'].get('embedding_text','')[:300]}"
        for c in top10
    )

    watched = filters.get("videos_already_watched") or []
    context = (
        f"Coach query: {query}\n"
        f"Athlete event: {filters.get('event')}\n"
        f"Weaknesses: {filters.get('weakness_tags')}\n"
        f"Age range: {filters.get('age_range')}\n"
        f"Weeks to competition: {filters.get('weeks_to_competition')}\n"
        f"Already watched: {watched}\n\n"
        f"Candidates:\n{candidate_text}"
    )

    response = openai_client.chat.completions.create(
        model=GPT4O_DEPLOYMENT,
        messages=[
            {"role": "system", "content": RERANKER_SYSTEM},
            {"role": "user",   "content": context},
        ],
        response_format={"type": "json_object"},
        max_tokens=600,
        temperature=0,
    )

    raw = response.choices[0].message.content
    # Handle both array and object wrapper
    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        scores = next(iter(parsed.values()))
    else:
        scores = parsed

    scores.sort(key=lambda x: x["relevance_score"], reverse=True)
    reranked = scores[:top_n]
    console.log(f"  Top-{top_n} after reranking:")
    for r in reranked:
        console.log(f"    {r['doc_id']} → score {r['relevance_score']}: {r['reason'][:80]}")
    return reranked


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 – SwimCoachGPT Generation (GPT-4 Turbo, streamed)
# ══════════════════════════════════════════════════════════════════════════════

SWIMCOACH_SYSTEM = """You are SwimCoachGPT, an elite AI swimming coach assistant.
You provide structured, evidence-based coaching recommendations grounded ONLY in
the retrieved knowledge base context provided. Always cite video IDs.

Format your response in EXACTLY these 5 sections:

## 1. Must-Watch Videos
List video IDs with rationale and athlete match score (1-10).

## 2. Videos to Avoid Now
List any videos that are too advanced, need prerequisites, or are not yet relevant.

## 3. Gap Analysis
Videos watched by similar athletes that this athlete has NOT yet watched.

## 4. Suggested Viewing Sequence
Ordered playlist with rationale for each step.

## 5. Confidence Level
State HIGH / MEDIUM / LOW and explain based on how many matching historical cases exist.

If the query is not related to swimming coaching, respond with:
"Sorry, I'm unable to give the information. Please re-phrase your query with respect to my domain knowledge."
"""

def generate_response(query: str, reranked: list[dict], candidates: list[dict],
                      filters: dict, stream: bool = True):
    """
    Stage 5: Generate the 5-section SwimCoachGPT response.
    Streams token-by-token if stream=True.
    """
    console.log("[bold cyan]Stage 5:[/] SwimCoachGPT generating response …\n")

    # Build context from top-5 reranked docs
    reranked_ids = {r["doc_id"] for r in reranked}
    context_docs = [c for c in candidates if c["id"] in reranked_ids]
    if not context_docs:
        context_docs = candidates[:5]

    context_text = "\n\n---\n\n".join(
        f"[{c['id']}]\n"
        f"Type: {c['metadata'].get('doc_type','')}\n"
        f"Tags: {', '.join(c['metadata'].get('topic_tags', []))}\n"
        f"Difficulty: {c['metadata'].get('difficulty','')}\n"
        f"Summary: {c['metadata'].get('embedding_text','')}"
        for c in context_docs
    )

    rerank_context = "\n".join(
        f"- {r['doc_id']} (score {r['relevance_score']}): {r['reason']}"
        for r in reranked
    )

    user_message = (
        f"Coach query: {query}\n\n"
        f"Athlete profile from query decomposition:\n{json.dumps(filters, indent=2)}\n\n"
        f"Reranker assessment:\n{rerank_context}\n\n"
        f"Retrieved knowledge base context:\n{context_text}"
    )

    if stream:
        full_response = ""
        with openai_client.chat.completions.stream(
            model=GPT4_TURBO_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SWIMCOACH_SYSTEM},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=1500,
            temperature=0.3,
        ) as stream_ctx:
            console.print("\n" + "─" * 70)
            console.print("[bold green]SwimCoachGPT Response[/]\n")
            for text in stream_ctx.text_stream:
                print(text, end="", flush=True)
                full_response += text
        print("\n" + "─" * 70 + "\n")
        return full_response
    else:
        response = openai_client.chat.completions.create(
            model=GPT4_TURBO_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SWIMCOACH_SYSTEM},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=1500,
            temperature=0.3,
        )
        result = response.choices[0].message.content
        console.print("\n" + "─" * 70)
        console.print("[bold green]SwimCoachGPT Response[/]\n")
        print(result)
        print("─" * 70 + "\n")
        return result


# ══════════════════════════════════════════════════════════════════════════════
# FULL RAG QUERY PIPELINE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_rag_query(query: str, stream: bool = True) -> str:
    """
    Orchestrate all 5 stages of the RAG query pipeline.
    Returns the final SwimCoachGPT response string.
    """
    console.print(Panel(
        f"[bold]Query:[/] {query}",
        title="SwimCoach RAG · Query Pipeline",
        style="blue",
    ))

    # Domain guard
    if not is_in_domain(query):
        console.print(f"\n[yellow]{OUT_OF_DOMAIN_REPLY}[/]\n")
        return OUT_OF_DOMAIN_REPLY

    # Stage 1: Decompose
    filters = decompose_query(query)

    # Stage 2: Parallel retrieval
    sem_hits, bm25_hits = parallel_retrieve(query, filters)

    # Stage 3: RRF fusion
    merged = rrf_fusion(sem_hits, bm25_hits, top_n=20)

    # Stage 4: LLM reranker
    reranked = llm_rerank(query, merged, filters, top_n=5)

    # Stage 5: Generate
    response = generate_response(query, reranked, merged, filters, stream=stream)
    return response


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwimCoach RAG — Query Pipeline")
    parser.add_argument(
        "--query", "-q",
        default=(
            "Am I missing any must-watch videos for this athlete profile? "
            "Athlete: 19M, 200m Butterfly, underwater dolphin kick weakness, "
            "6 weeks to State Championships."
        ),
        help="Natural language coaching query",
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Disable streaming (return full response at once)",
    )
    args = parser.parse_args()
    run_rag_query(args.query, stream=not args.no_stream)
