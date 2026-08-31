import json
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from config import settings

PROMPT_VERSION = "v0-baseline"
MODEL = "claude-sonnet-5"
SYSTEM_PROMPT = (
    "You are a soccer facts assistant. If you are not confident in a specific, "
    "verifiable detail (an exact scoreline, date, transfer fee, or similar), "
    "set abstained to true and briefly say what you're unsure of, rather than "
    "guessing. Only answer directly when you're confident the fact is correct."
)
RUN=3


class QAAnswer(BaseModel):
    answer: str = Field(
        description="Your direct answer. If abstained is True, briefly explain "
        "what you're unsure of instead of guessing a specific value."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description='Your self-reported confidence: "high", "medium", or "low". '
        'Use "low" whenever abstained is True.'
    )
    abstained: bool = Field(
        description="True if you are not confident in a specific, verifiable "
        "fact (exact scoreline, date, fee, etc.) and are declining "
        "to guess rather than risk an incorrect answer. False if "
        "you are giving a direct, confident answer."
    )


client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


def load_golden_dataset(path: str = "golden_dataset.jsonl"):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ask(question: str, model: str = MODEL, system: str | None = None) -> QAAnswer:
    kwargs = {
        "model": model,
        "max_tokens": 2024,
        "messages": [{"role": "user", "content": question}],
        "output_format": QAAnswer,
    }
    if system:
        kwargs["system"] = system
    response = client.messages.parse(**kwargs)
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
            "abstained": model_answer.abstained,
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "run": RUN,
        }
    )

OUTPUT_DIR = Path(f"results")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / f"qa_results_{MODEL}_{PROMPT_VERSION}_R{RUN}.jsonl"
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.writelines(json.dumps(result) + "\n" for result in results)
