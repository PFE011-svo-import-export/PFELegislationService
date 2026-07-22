import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate, EvaluationDataset

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
)

from dotenv import load_dotenv
load_dotenv()
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

TEST_SET_PATH = "ragas_dataset.json"

# Claude acts as LLM as judge;
claude_llm = LangchainLLMWrapper(
    ChatAnthropic(model="claude-sonnet-4-6", max_tokens=4096, temperature=0)
)

# Embeddings come from OpenAI, same as project
embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small")
)

with open(TEST_SET_PATH, encoding="utf-8") as f:
    data = json.load(f)

ragas_data = [
    {
        "user_input": item["question"],
        "retrieved_contexts": item["contexts"],
        "response": item["answer"],
        "reference": item["ground_truth"],
    }

    for item in data
]

dataset = Dataset.from_list(ragas_data)

evaluation_dataset = EvaluationDataset.from_hf_dataset(dataset)

metrics = [
    Faithfulness(),
    AnswerRelevancy(),
    AnswerCorrectness(),
    ContextPrecision(),
    ContextRecall(),
]

results = evaluate(evaluation_dataset, metrics=metrics, llm=claude_llm, embeddings=embeddings)

df = results.to_pandas()
print(df)