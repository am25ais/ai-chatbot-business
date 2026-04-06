from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from supabase import create_client
from anthropic import Anthropic
from dotenv import load_dotenv
import hashlib

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class ChatRequest(BaseModel):
    message: str
    business_id: str

def simple_embedding(text):
    # Simple deterministic embedding using hashing
    words = text.lower().split()
    embedding = [0.0] * 384
    for i, word in enumerate(words):
        hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
        for j in range(min(10, 384)):
            embedding[(hash_val + j * i) % 384] += 1.0
    # Normalize
    magnitude = sum(x**2 for x in embedding) ** 0.5
    if magnitude > 0:
        embedding = [x/magnitude for x in embedding]
    return embedding

def search_documents(query, business_id):
    query_embedding = simple_embedding(query)
    result = supabase.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_business_id": business_id,
        "match_count": 3
    }).execute()
    return [r["content"] for r in result.data]

@app.get("/")
def home():
    return {"status": "Chatbot API is running!"}

@app.post("/chat")
def chat(request: ChatRequest):
    relevant_docs = search_documents(request.message, request.business_id)
    context = "\n".join(relevant_docs)
    message = anthropic.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=f"You are a helpful customer support agent. Answer based on this business information only:\n\n{context}",
        messages=[{"role": "user", "content": request.message}]
    )
    return {"response": message.content[0].text}