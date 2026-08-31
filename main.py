import json

import anthropic
from pydantic import BaseModel

from config import settings

PROMPT_VERSION = "v1"
MODEL = "claude-haiku-4-5"
SYSTEM_PROMPT = 'You are a soccer facts assistant. Only answer with facts you are confident are correct. If you are not certain of an exact detail (a scoreline, date, transfer fee, or similar specific fact), say so explicitly rather than guessing, and set confidence to "low".'


class QAAnswer(BaseModel):
    answer: str
    confidence: str


client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


def load_golden_dataset(path: str = "golden_dataset.jsonl"):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ask(question: str, model: str = MODEL) -> QAAnswer:
    response = client.messages.parse(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": question},
        ],
        output_format=QAAnswer,
    )
    return response.parsed_output


results = []
for row in load_golden_dataset():
    model_answer = ask(row["question"])
    results.append(
        {
            "question": row["question"],
            "golden_answer": row["answer"],
            "model_answer": model_answer.answer,
            "confidence": model_answer.confidence,
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
        }
    )

OUTPUT_PATH = f"/results/qa_results_{MODEL}_{PROMPT_VERSION}.jsonl"
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.writelines(json.dumps(result) + "\n" for result in results)
