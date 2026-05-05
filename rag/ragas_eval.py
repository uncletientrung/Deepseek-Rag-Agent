from datasets import Dataset
from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)

from ragas.llms import LangchainLLMWrapper

from rag.llm import get_llm  
import os
os.environ["OPENAI_API_KEY"] = "disabled" 

def get_ragas_llm():
    llm = get_llm()  
    return LangchainLLMWrapper(llm)


ragas_llm = get_ragas_llm()
def evaluate_ragas(question: str, answer: str, contexts: list):
    if not question or not answer:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0
        }

    if not contexts:
        contexts = [""]

    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts]
    })

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy
        ],
        llm=ragas_llm
    )

    df = result.to_pandas()

    return {
        "faithfulness": float(df["faithfulness"][0]),
    }