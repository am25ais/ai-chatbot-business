from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from sentence_transformers import SentenceTransformer
from supabase import create_client
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
print("Loading models...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class ChatRequest(BaseModel):
    message: str
    business_id: str

def search_documents(query, business_id):
    query_embedding = embedder.encode(query).tolist()
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