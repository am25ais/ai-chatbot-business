import os
from sentence_transformers import SentenceTransformer
from supabase import create_client
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Load free embedding model
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Step 1 — Embed and store business data
def store_business_data(business_id, documents):
    for doc in documents:
        embedding = embedder.encode(doc).tolist()
        supabase.table("documents").insert({
            "business_id": business_id,
            "content": doc,
            "embedding": embedding
        }).execute()
        print(f"✅ Stored: {doc[:60]}...")

# Step 2 — Search similar documents
def search_documents(query, business_id):
    query_embedding = embedder.encode(query).tolist()
    result = supabase.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_business_id": business_id,
        "match_count": 3
    }).execute()
    return [r["content"] for r in result.data]

# Step 3 — Answer using Claude
def ask_chatbot(question, business_id):
    relevant_docs = search_documents(question, business_id)
    context = "\n".join(relevant_docs)
    message = anthropic.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=f"You are a helpful customer support agent. Answer based on this business information only:\n\n{context}",
        messages=[{"role": "user", "content": question}]
    )
    return message.content[0].text

# --- TEST IT ---
business_data = [
    "We are Mario's Pizza. We are open 9am to 10pm every day.",
    "We deliver within 5 miles. Delivery takes 30-45 minutes.",
    "Our most popular pizza is Pepperoni Supreme for £14.99.",
    "We have gluten free bases available for £2 extra.",
    "To order call 555-1234 or visit our website.",
    "We are located at 123 Main Street, London.",
    "We accept cash, card and all digital payments.",
]

print("Storing business data...")
store_business_data("marios-pizza", business_data)

print("\nTesting chatbot...")
questions = [
    "What time do you close?",
    "Do you deliver?",
    "What's your most popular pizza?",
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {ask_chatbot(q, 'marios-pizza')}")