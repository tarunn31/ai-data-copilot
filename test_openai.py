import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ API key not found. Check your .env file.")
else:
    print("✅ API key loaded.")

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": "Say: My AI Copilot is working!"}
    ],
)

print("AI Response:")
print(response.choices[0].message.content)
