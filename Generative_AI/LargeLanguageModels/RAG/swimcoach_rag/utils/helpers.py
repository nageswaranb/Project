"""
utils/helpers.py
Shared utility functions: embeddings, retries, logging.
"""

import time
import base64
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIError
from rich.console import Console
from config.settings import openai_client, ADA_DEPLOYMENT

console = Console()


# ── Embedding ─────────────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
)
def embed_text(text: str) -> list[float]:
    """
    Convert text → 1536-dim Ada-002 embedding vector.
    Retries automatically on rate-limit / transient errors.
    """
    response = openai_client.embeddings.create(
        model=ADA_DEPLOYMENT,
        input=text,
    )
    return response.data[0].embedding


# ── Image helpers ─────────────────────────────────────────────────────────────

def image_to_base64(image_path: str) -> str:
    """Read a JPEG/PNG file and return base64-encoded string for GPT-4o Vision."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── GPT-4o Vision caption ─────────────────────────────────────────────────────

FRAME_CAPTION_SYSTEM = """You are an expert swimming coach and video analyst.
Caption each video frame in 1-3 sentences. Focus on:
- Stroke phase (catch, pull, push, recovery)
- Body position (head, hips, arms, legs alignment)
- Kick mechanics (frequency, amplitude, dolphin kick depth)
- Any visible coaching annotations or lane markers
Be precise and use technical swimming terminology."""

@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def caption_frame(image_path: str, deployment: str) -> str:
    """Send a single JPEG frame to GPT-4o Vision and return a 1-3 sentence caption."""
    b64 = image_to_base64(image_path)
    response = openai_client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": FRAME_CAPTION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "low",   # cost-efficient; use "high" for elite detail
                        },
                    },
                    {"type": "text", "text": "Describe this swimming video frame for coaching analysis."},
                ],
            },
        ],
        max_tokens=200,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


# ── GPT-4o summarise captions → final embedding_text ─────────────────────────

SUMMARY_SYSTEM = """You are a swimming coaching knowledge-base curator.
Given a list of per-frame captions from a coaching video, synthesise them into
a single coherent summary of 10-200 words that captures:
- The primary technique or drill demonstrated
- Camera angle and underwater perspective (if present)
- Skill level (beginner / intermediate / advanced / elite)
- Key coaching focus points
This summary will be used for semantic search retrieval. Be dense with
domain-specific keywords (stroke name, kick type, phase names)."""

@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def summarise_captions(captions: list[str], deployment: str) -> str:
    """Collapse all per-frame captions into a final 10-200 word embedding_text."""
    joined = "\n".join(f"- {c}" for c in captions)
    response = openai_client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": f"Frame captions:\n{joined}\n\nWrite the final video summary:"},
        ],
        max_tokens=300,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
