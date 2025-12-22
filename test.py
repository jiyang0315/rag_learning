import os
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("OPENAI_BASE_URL", "https://api.road2all.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY 未设置")

print("API_BASE:", API_BASE)
print("API_KEY :", API_KEY[:8], "...")

# 关键：用 api_base（不是 base_url）
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, Document
from llama_index.core.evaluation import DatasetGenerator

Settings.llm = OpenAI(
    model="gpt-4o-mini",
    api_key=API_KEY,
    api_base=API_BASE,   # ✅ 强制走你的网关
)

docs = [Document(text="SliderSR is an image super-resolution project based on deep learning.")]
gen = DatasetGenerator.from_documents(docs)

print("🚀 开始生成评估问题...")
qs = gen.generate_questions_from_nodes(num=3)

print("🎉 生成结果：")
for i, q in enumerate(qs, 1):
    print(i, q)
