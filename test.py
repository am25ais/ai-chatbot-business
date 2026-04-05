import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

# Check if key is being loaded
api_key = os.getenv("ANTHROPIC_API_KEY")
print(f"Key loaded: {api_key[:20] if api_key else 'NOT FOUND'}")

client = anthropic.Anthropic(api_key=api_key)

message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    system="You are a helpful customer support agent for Mario's Pizza. We are open 9am-10pm. We deliver within 5 miles. Our best seller is the Pepperoni Supreme for $14.99.",
    messages=[
        {"role": "user", "content": "What time do you close?"}
    ]
)

print(message.content[0].text)