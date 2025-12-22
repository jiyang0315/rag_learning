"""
scripts/2_choose_chunk_size.py

评估不同 chunk_size 下 RAG 系统的：
- 平均响应时间
- Faithfulness
- Relevancy

✅ 统一使用 OpenAI-compatible 网关（road2all）：
- LLM: gpt-4o-mini
- Embedding: text-embedding-3-small（如网关不支持请换成它支持的 embedding 模型名）
"""

import os
import random
import time

import nest_asyncio
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.core.evaluation import DatasetGenerator, FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.core.node_parser import SentenceSplitter

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 解决某些环境下 asyncio 嵌套问题（脚本里不一定需要，但保留无害）
nest_asyncio.apply()

# ========== 1) 读取环境变量 ==========
load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.road2all.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "REMOVED_SECRET")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 没有设置（.env 或环境变量）")

# 也写入环境（有些组件会读）
os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

print(f"✅ Using OPENAI_BASE_URL: {OPENAI_BASE_URL}")
print(f"✅ Using OPENAI_API_KEY : {OPENAI_API_KEY[:8]}...")

# ========== 2) 全局统一设置 LLM & Embedding（避免走官方导致401） ==========
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    api_base=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)

# ⚠️ 如果 road2all 不支持 text-embedding-3-small，请换成它支持的 embedding 模型名
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_base=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)

# ========== 3) 评估函数 ==========
def evaluate_response_time_and_accuracy(
    chunk_size: int,
    eval_questions: list[str],
    eval_documents,
    faithfulness_evaluator: FaithfulnessEvaluator,
    relevancy_evaluator: RelevancyEvaluator,
):
    total_response_time = 0.0
    total_faithfulness = 0.0
    total_relevancy = 0.0

    # ✅ overlap 必须 < chunk_size
    chunk_overlap = min(50, max(0, chunk_size // 4))
    
    # 把原始长文档，按“句子边界”为优先，切成一块一块不超过 chunk_size 的文本（chunk），相邻 chunk 之间保留 chunk_overlap 的重叠内容。
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 构建向量索引（这里会触发 embedding）
    vector_index = VectorStoreIndex.from_documents(eval_documents, transformations=[splitter])

    # query engine
    query_engine = vector_index.as_query_engine(similarity_top_k=5)
    num_questions = len(eval_questions)

    for question in eval_questions:
        start_time = time.time()
        response = query_engine.query(question)
        elapsed_time = time.time() - start_time

        # passing 是 bool，这里累加会自动变成 0/1
        faithfulness_pass = faithfulness_evaluator.evaluate_response(response=response).passing
        relevancy_pass = relevancy_evaluator.evaluate_response(query=question, response=response).passing

        total_response_time += elapsed_time
        total_faithfulness += float(faithfulness_pass)
        total_relevancy += float(relevancy_pass)

    return (
        total_response_time / num_questions,
        total_faithfulness / num_questions,
        total_relevancy / num_questions,
    )

# ========== 4) 主类 ==========
class RAGEvaluator:
    def __init__(self, data_dir: str, num_eval_questions: int, chunk_sizes: list[int]):
        self.data_dir = data_dir
        self.num_eval_questions = num_eval_questions
        self.chunk_sizes = chunk_sizes

        self.documents = self.load_documents()

        # 生成评估问题（这里也会调用 LLM）
        self.eval_questions = self.generate_eval_questions()
        print("✅ eval_questions sample:", self.eval_questions[:3])

        # 评估器统一用同一个 LLM（Settings.llm 也行；这里显式传更清晰）
        self.llm_eval = OpenAI(
            model="gpt-4o-mini",
            api_base=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
        )

        self.faithfulness_evaluator = self.create_faithfulness_evaluator()
        self.relevancy_evaluator = self.create_relevancy_evaluator()

    def load_documents(self):
        return SimpleDirectoryReader(self.data_dir).load_data()

    def generate_eval_questions(self):
        eval_documents = self.documents[:20]
        data_generator = DatasetGenerator.from_documents(eval_documents)

        # num 参数不同版本可能不支持；你当前环境支持 generate_questions_from_nodes(num=...)
        eval_questions = data_generator.generate_questions_from_nodes(num=max(self.num_eval_questions, 10))
        return random.sample(eval_questions, self.num_eval_questions)

    
    # FaithfulnessEvaluator 判断的不是“你说得对不对”，而是“你是不是老实，只说了我给你的内容”。
    def create_faithfulness_evaluator(self):
        faithfulness_evaluator = FaithfulnessEvaluator(llm=self.llm_eval)

        # 可选：自定义 prompt（你这里的 key 可能不是正确的 prompt key；先不强行改，避免无效）
        # 如果你确实要改 prompt，建议先打印 faithfulness_evaluator.get_prompts().keys()
        faithfulness_new_prompt_template = PromptTemplate(
        """
        Please tell if a given piece of information is directly supported by the context.
        You need to answer with either YES or NO.
        Answer YES if any part of the context explicitly supports the information, even if most of the context is unrelated.
        If the context does not explicitly support the information, answer NO.
        """
        )
        # 如果你不确定 key，先注释掉下一行，避免 silent no-op
        # faithfulness_evaluator.update_prompts({"your_prompt_key": faithfulness_new_prompt_template})

        return faithfulness_evaluator

    # RelevancyEvaluator 用来判断：模型“回答的内容是否真正回答了用户的问题”
    def create_relevancy_evaluator(self):
        return RelevancyEvaluator(llm=self.llm_eval)

    def run(self):
        for chunk_size in self.chunk_sizes:
            avg_time, avg_faith, avg_rel = evaluate_response_time_and_accuracy(
                chunk_size=chunk_size,
                eval_questions=self.eval_questions,
                eval_documents=self.documents[:20],
                faithfulness_evaluator=self.faithfulness_evaluator,
                relevancy_evaluator=self.relevancy_evaluator,
            )

            print(
                f"Chunk size {chunk_size} | "
                f"Avg time: {avg_time:.2f}s | "
                f"Faithfulness: {avg_faith:.2f} | "
                f"Relevancy: {avg_rel:.2f}"
            )

# ========== 5) 参数解析 ==========
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="RAG Chunk Size Evaluation")
    parser.add_argument("--data_dir", type=str, default="../data", help="Directory of the documents")
    parser.add_argument("--num_eval_questions", type=int, default=25, help="Number of evaluation questions")
    parser.add_argument("--chunk_sizes", nargs="+", type=int, default=[128, 256], help="List of chunk sizes")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluator = RAGEvaluator(
        data_dir=args.data_dir,
        num_eval_questions=args.num_eval_questions,
        chunk_sizes=args.chunk_sizes,
    )
    evaluator.run()
