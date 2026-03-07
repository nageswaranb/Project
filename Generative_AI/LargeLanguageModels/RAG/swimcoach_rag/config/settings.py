"""
config/settings.py
Loads all environment variables and initialises shared Azure / Pinecone clients.
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from pinecone import Pinecone, ServerlessSpec
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField,
    SearchFieldDataType, VectorSearch,
    HnswAlgorithmConfiguration, VectorSearchProfile,
    SearchField,
)
from azure.core.credentials import AzureKeyCredential
from azure.cosmos import CosmosClient

load_dotenv()

# ── Azure OpenAI ──────────────────────────────────────────────────────────────
openai_client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
)

GPT4O_DEPLOYMENT      = os.environ.get("AZURE_OPENAI_GPT4O_DEPLOYMENT",      "gpt-4o")
GPT4_TURBO_DEPLOYMENT = os.environ.get("AZURE_OPENAI_GPT4_TURBO_DEPLOYMENT", "gpt-4-turbo")
ADA_DEPLOYMENT        = os.environ.get("AZURE_OPENAI_ADA_DEPLOYMENT",        "text-embedding-ada-002")

# ── Pinecone ──────────────────────────────────────────────────────────────────
_pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "swimcoach-kb")
PINECONE_DIMENSION  = 1536   # ada-002 output dimension

def get_pinecone_index():
    """Return (creating if needed) the Pinecone serverless index."""
    existing = [i.name for i in _pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        _pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.environ.get("PINECONE_CLOUD", "aws"),
                region=os.environ.get("PINECONE_REGION", "us-east-1"),
            ),
        )
        print(f"[Pinecone] Created index '{PINECONE_INDEX_NAME}'")
    return _pc.Index(PINECONE_INDEX_NAME)

# ── Azure Cognitive Search (BM25) ─────────────────────────────────────────────
SEARCH_ENDPOINT   = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_API_KEY    = os.environ["AZURE_SEARCH_API_KEY"]
SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "swimcoach-bm25")

_search_credential = AzureKeyCredential(SEARCH_API_KEY)

def get_search_client() -> SearchClient:
    return SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX_NAME,
        credential=_search_credential,
    )

def ensure_search_index():
    """Create the BM25 search index if it doesn't already exist."""
    idx_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=_search_credential)
    existing = [i.name for i in idx_client.list_indexes()]
    if SEARCH_INDEX_NAME in existing:
        return

    fields = [
        SimpleField(name="id",           type=SearchFieldDataType.String, key=True),
        SimpleField(name="video_id",     type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="doc_type",     type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="difficulty",   type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="embedding_text", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SearchableField(name="topic_tags_str", type=SearchFieldDataType.String),
    ]
    idx = SearchIndex(name=SEARCH_INDEX_NAME, fields=fields)
    idx_client.create_index(idx)
    print(f"[ACS] Created BM25 index '{SEARCH_INDEX_NAME}'")

# ── Cosmos DB ─────────────────────────────────────────────────────────────────
def get_cosmos_container():
    client = CosmosClient(
        url=os.environ["COSMOS_ENDPOINT"],
        credential=os.environ["COSMOS_KEY"],
    )
    db = client.get_database_client(os.environ.get("COSMOS_DATABASE", "swimcoach"))
    return db.get_container_client(os.environ.get("COSMOS_CONTAINER", "coaching_logs"))
