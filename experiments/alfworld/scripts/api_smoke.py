import os

from openai import OpenAI


client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv("OPENAI_BASE_URL", "https://spacetimeai.cc/v1"),
)
model = os.getenv("OPENAI_MODEL", "gpt-6-luna")

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
)

print(f"model={model}")
print(response.choices[0].message.content)
