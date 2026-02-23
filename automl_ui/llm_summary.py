import json
from pathlib import Path
from groq import Groq
from api_keys import *

client = Groq()


def load_all_jsonl(data_dir):
    text = []
    for p in Path(data_dir).glob("*.jsonl"):
        content = p.read_text(encoding="utf-8")
        text.append(f"\nFILE: {p.name}\n{content}")
    return "\n".join(text)


def generate_summary(data_dir):

    context = load_all_jsonl(data_dir)

    prompt = f"""
You are a senior ML auditor.

Analyze this AutoML pipeline execution.

Explain:
- dataset understanding
- feature engineering
- model selection reasoning
- training quality
- evaluation quality
- risks
- production readiness

DATA:
{context}
"""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content