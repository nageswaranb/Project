from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(title="RAG Tool Agent", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models nvhgvh ─────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []   # [{"role": "user"|"assistant", "content": "..."}]

class ChatResponse(BaseModel):
    answer: str

# ── Build the agent once at startup ───────────────────────────
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512
)

vectorstore = PineconeVectorStore(
    index_name=os.environ["PINECONE_INDEX"],   # set in .env / HF Secrets
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

@tool
def document_search(query: str) -> str:
    """Search and retrieve relevant information from uploaded documents."""
    docs = retriever.invoke(query)
    return "\n\n".join(
        f"Source {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

agent = create_react_agent(
    model=llm,
    tools=[document_search],
    state_modifier="You are an educational assistant. Answer questions based on the story."
)

# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "RAG Tool Agent is running"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = []
    for msg in req.history[-8:]:   # keep last 4 exchanges
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))

    messages = history + [HumanMessage(content=req.question)]
    result = agent.invoke({"messages": messages})
    answer = result["messages"][-1].content
    return ChatResponse(answer=answer)
