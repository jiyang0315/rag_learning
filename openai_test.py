import os
from openai import OpenAI

base = os.getenv("OPENAI_BASE_URL","https://api.road2all.com/v1")
key  = os.getenv("OPENAI_API_KEY", "REMOVED_SECRET")
print("BASE:", base)
print("KEY :", (key or "")[:8])

client = OpenAI(base_url=base, api_key=key)
r = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[{"role":"user","content":"ping"}]
)
print(r.choices[0].message.content)

