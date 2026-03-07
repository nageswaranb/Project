"""
ingestion/pipeline.py
─────────────────────────────────────────────────────────────────
DATA INGESTION PIPELINE
  Video Upload → Azure Blob → ffmpeg Chunker → OpenCV Frame Extractor
  → GPT-4o Vision Captioner → GPT-4o Summariser → Ada-002 Embedder
  → Pinecone (semantic) + Azure Cognitive Search (BM25) + Cosmos DB

Run standalone:
    python -m ingestion.pipeline --video path/to/video.mp4 --video-id V-0999

Seed KB with sample data (no video needed):
    python -m ingestion.pipeline --seed
─────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import argparse
import tempfile
import subprocess
from pathlib import Path

import cv2
import ffmpeg
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel

# ── Make package importable from project root ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import (
    openai_client,
    GPT4O_DEPLOYMENT,
    get_pinecone_index,
    ensure_search_index,
    get_search_client,
    get_cosmos_container,
)
from utils.helpers import embed_text, caption_frame, summarise_captions
from kb.seed_data import COACHING_LOGS, VIDEO_CATALOG, ATHLETE_PROFILES

console = Console()

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE_MB   = 20
FRAME_INTERVAL  = 2.5   # seconds between sampled frames
MAX_FRAMES_CHUNK = 40   # cap frames per chunk to control API costs


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 – ffmpeg: split video into ≤20 MB chunks
# ══════════════════════════════════════════════════════════════════════════════

def chunk_video(video_path: str, output_dir: str) -> list[str]:
    """
    Split video into segments of approximately CHUNK_SIZE_MB each.
    Returns list of chunk file paths (sorted).
    """
    console.log(f"[bold cyan]Stage 1:[/] Chunking video → {CHUNK_SIZE_MB} MB segments")

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    n_chunks = max(1, int(file_size_mb // CHUNK_SIZE_MB) + 1)

    # Probe duration
    probe = ffmpeg.probe(str(video_path))
    duration = float(probe["format"]["duration"])
    segment_time = duration / n_chunks

    pattern = str(output_dir / "chunk_%03d.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(segment_time),
        "-reset_timestamps", "1",
        pattern,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    chunks = sorted(output_dir.glob("chunk_*.mp4"))
    console.log(f"  ✓ {len(chunks)} chunks created from {file_size_mb:.1f} MB video")
    return [str(c) for c in chunks]


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 – OpenCV: extract 1 frame per 2-3 s from each chunk
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames(chunk_path: str, output_dir: str) -> list[str]:
    """
    Sample 1 frame every FRAME_INTERVAL seconds from a video chunk.
    Returns list of JPEG file paths.
    """
    cap = cv2.VideoCapture(chunk_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    skip = int(fps * FRAME_INTERVAL)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths, frame_idx, saved = [], 0, 0
    while saved < MAX_FRAMES_CHUNK:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip == 0:
            out_path = str(output_dir / f"frame_{saved:04d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_paths.append(out_path)
            saved += 1
        frame_idx += 1
    cap.release()
    return frame_paths


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 – GPT-4o Vision: caption each frame
# ══════════════════════════════════════════════════════════════════════════════

def caption_all_frames(frame_paths: list[str]) -> list[str]:
    """Send every frame to GPT-4o Vision; return list of captions."""
    captions = []
    for fp in tqdm(frame_paths, desc="  Captioning frames", unit="frame"):
        caption = caption_frame(fp, GPT4O_DEPLOYMENT)
        captions.append(caption)
    return captions


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 – GPT-4o: summarise all captions → final embedding_text
# ══════════════════════════════════════════════════════════════════════════════

def build_embedding_text(all_captions: list[str]) -> str:
    """Combine all per-frame captions into a single 10-200 word summary."""
    console.log("[bold cyan]Stage 4:[/] Summarising captions → embedding_text")
    return summarise_captions(all_captions, GPT4O_DEPLOYMENT)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 – Embed + Index (Pinecone + ACS + Cosmos)
# ══════════════════════════════════════════════════════════════════════════════

def index_document(doc: dict, embedding_text: str):
    """
    Embed embedding_text and upsert into:
      - Pinecone (semantic vector search)
      - Azure Cognitive Search (BM25 keyword search)
      - Cosmos DB (metadata / JSON store)
    """
    console.log("[bold cyan]Stage 5:[/] Embedding + indexing document")

    # 5a. Embed
    vector = embed_text(embedding_text)
    doc_id = doc.get("video_id") or doc.get("log_id") or doc.get("athlete_id")

    # 5b. Pinecone upsert
    pinecone_index = get_pinecone_index()
    pinecone_index.upsert(
        vectors=[{
            "id": doc_id,
            "values": vector,
            "metadata": {
                "doc_type":       doc.get("doc_type", "unknown"),
                "video_id":       doc.get("video_id", ""),
                "topic_tags":     doc.get("topic_tags", []),
                "difficulty":     doc.get("difficulty", ""),
                "embedding_text": embedding_text[:500],  # Pinecone metadata cap
            },
        }]
    )
    console.log(f"  ✓ Pinecone upserted → {doc_id}")

    # 5c. Azure Cognitive Search upsert
    ensure_search_index()
    search_client = get_search_client()
    search_client.upload_documents(documents=[{
        "id":             doc_id.replace("-", "_"),   # ACS key must be alphanumeric/_
        "video_id":       doc.get("video_id", ""),
        "doc_type":       doc.get("doc_type", ""),
        "difficulty":     doc.get("difficulty", ""),
        "embedding_text": embedding_text,
        "topic_tags_str": " ".join(doc.get("topic_tags", [])),
    }])
    console.log(f"  ✓ ACS BM25 indexed  → {doc_id}")

    # 5d. Cosmos DB upsert (full metadata)
    try:
        container = get_cosmos_container()
        cosmos_doc = {**doc, "id": doc_id, "embedding_text": embedding_text}
        container.upsert_item(cosmos_doc)
        console.log(f"  ✓ Cosmos DB stored  → {doc_id}")
    except Exception as e:
        console.log(f"  [yellow]Cosmos DB skipped (optional): {e}[/]")


# ══════════════════════════════════════════════════════════════════════════════
# FULL VIDEO INGESTION ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def ingest_video(video_path: str, video_id: str, topic_tags: list[str] = None,
                 difficulty: str = "intermediate"):
    """
    End-to-end pipeline: raw video file → vector store.
    Stages: chunk → extract frames → caption → summarise → embed → index
    """
    console.print(Panel(
        f"[bold]Ingesting Video:[/] {video_path}\n[bold]ID:[/] {video_id}",
        title="SwimCoach RAG · Data Ingestion Pipeline",
        style="blue",
    ))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Stage 1: chunk
        chunks = chunk_video(video_path, str(tmp / "chunks"))

        # Stages 2 + 3: per-chunk frame extraction + captioning
        all_captions = []
        for i, chunk in enumerate(chunks):
            console.log(f"[bold cyan]Stage 2-3:[/] Chunk {i+1}/{len(chunks)} — extract & caption")
            frames = extract_frames(chunk, str(tmp / f"frames_{i}"))
            console.log(f"  {len(frames)} frames extracted from chunk {i+1}")
            captions = caption_all_frames(frames)
            all_captions.extend(captions)

        # Stage 4: summarise
        embedding_text = build_embedding_text(all_captions)
        console.log(f"\n[green]embedding_text ({len(embedding_text.split())} words):[/]\n{embedding_text}\n")

        # Stage 5: index
        doc = {
            "video_id":  video_id,
            "doc_type":  "video_catalog",
            "topic_tags": topic_tags or [],
            "difficulty": difficulty,
            "embedding_text": embedding_text,
        }
        index_document(doc, embedding_text)

    console.print(Panel(f"[bold green]✓ Ingestion complete for {video_id}[/]", style="green"))


# ══════════════════════════════════════════════════════════════════════════════
# SEED KB (no video — use pre-written embedding_text from seed_data.py)
# ══════════════════════════════════════════════════════════════════════════════

def seed_knowledge_base():
    """
    Bootstrap the KB by indexing all seed documents.
    Uses the pre-written embedding_text directly (no video processing needed).
    """
    console.print(Panel(
        "Seeding Knowledge Base with sample coaching data",
        title="SwimCoach RAG · KB Seed",
        style="cyan",
    ))

    all_docs = []

    # Video catalog (have embedding_text already)
    for v in VIDEO_CATALOG:
        all_docs.append((v, v["embedding_text"]))

    # Coaching logs (build embedding_text from structured fields)
    for log in COACHING_LOGS:
        ap = log["athlete_profile"]
        et = (
            f"Coaching log {log['log_id']}. "
            f"Athlete: {ap['age']}yr {ap['gender']} {ap['event']}. "
            f"Weaknesses: {', '.join(ap['weakness'])}. "
            f"Videos recommended: {', '.join(v['video_id'] for v in log['videos_recommended'])}. "
            f"Performance improvement: {log['performance_delta']['improvement_pct']}%. "
            f"Notes: {log.get('notes', '')}"
        )
        all_docs.append((log, et))

    # Athlete profiles
    for ath in ATHLETE_PROFILES:
        all_docs.append((ath, ath["embedding_text"]))

    console.log(f"Indexing {len(all_docs)} documents …")
    for doc, et in tqdm(all_docs, desc="Indexing"):
        try:
            index_document(doc, et)
        except Exception as e:
            console.log(f"[red]Error indexing {doc}:[/] {e}")

    console.print(Panel(
        f"[bold green]✓ Seeded {len(all_docs)} documents into Pinecone + ACS + Cosmos[/]",
        style="green",
    ))


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwimCoach RAG — Data Ingestion Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",    help="Path to video file (.mp4/.mov/.avi)")
    group.add_argument("--seed",     action="store_true", help="Seed KB with sample data")
    parser.add_argument("--video-id",   default="V-0001", help="Video ID (used with --video)")
    parser.add_argument("--tags",       nargs="*",        help="Topic tags  (used with --video)")
    parser.add_argument("--difficulty", default="intermediate", help="Difficulty level")
    args = parser.parse_args()

    if args.seed:
        seed_knowledge_base()
    else:
        ingest_video(args.video, args.video_id, args.tags, args.difficulty)
