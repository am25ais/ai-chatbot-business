import os
import hashlib
from supabase import create_client
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Same embedding function as server.py
def simple_embedding(text):
    words = text.lower().split()
    embedding = [0.0] * 384
    for i, word in enumerate(words):
        hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
        for j in range(min(10, 384)):
            embedding[(hash_val + j * i) % 384] += 1.0
    magnitude = sum(x**2 for x in embedding) ** 0.5
    if magnitude > 0:
        embedding = [x/magnitude for x in embedding]
    return embedding

# Step 1 — Clear old data
def clear_business_data(business_id):
    supabase.table("documents").delete().eq("business_id", business_id).execute()
    print(f"✅ Cleared old data for {business_id}")

# Step 2 — Store new data
def store_business_data(business_id, documents):
    for doc in documents:
        embedding = simple_embedding(doc)
        supabase.table("documents").insert({
            "business_id": business_id,
            "content": doc,
            "embedding": embedding
        }).execute()
        print(f"✅ Stored: {doc[:60]}...")

# Step 3 — Search
def search_documents(query, business_id):
    query_embedding = simple_embedding(query)
    result = supabase.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_business_id": business_id,
        "match_count": 3
    }).execute()
    return [r["content"] for r in result.data]

# Step 4 — Ask chatbot
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

# --- RUN IT ---
business_data = [
    "We are Mario's Pizza. We are open 9am to 10pm every day.",
    "We deliver within 5 miles. Delivery takes 30-45 minutes.",
    "Our most popular pizza is Pepperoni Supreme for £14.99.",
    "We have gluten free bases available for £2 extra.",
    "To order call 555-1234 or visit our website.",
    "We are located at 123 Main Street, London.",
    "We accept cash, card and all digital payments.",
]

print("Clearing old data...")
clear_business_data("marios-pizza")

print("\nStoring new data...")
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