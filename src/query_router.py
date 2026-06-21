from dataclasses import dataclass
import json
from ollama_client import chat


@dataclass
class QueryRoute:
    intent: str
    rewritten_query: str
    document_id: int | None = None
    jurisdiction: str | None = None
    authority: str | None = None
    confidence: float = 0.0


def classify_query(question, metadata, config):
    docs_preview = "\n".join(
        f"- ID {doc_id}: {m['official_name']} | "
        f"Authority: {m.get('authority', '')} | "
        f"Jurisdiction: {m.get('jurisdiction', '')}"
        for doc_id, m in list(metadata.items())[:80]
    )

    prompt = f"""
You are a query router for an AI governance RAG system.

Classify the user's question and return ONLY valid JSON.

Allowed intents:
- definition
- obligation
- risk_classification
- enforcement
- comparison
- document_specific
- jurisdiction_specific
- general

Rules:
- Only choose document_id if the question clearly asks about a specific document.
- Only choose jurisdiction if the question clearly asks about a jurisdiction.
- If unsure, leave document_id, jurisdiction, and authority as null.
- Rewrite the query to improve retrieval.
- Confidence must be between 0 and 1.

Available documents:
{docs_preview}

Question:
{question}

Return JSON with this shape:
{{
  "intent": "...",
  "rewritten_query": "...",
  "document_id": null,
  "jurisdiction": null,
  "authority": null,
  "confidence": 0.0
}}
"""

    raw = chat(prompt, config)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return QueryRoute(
            intent="general",
            rewritten_query=question,
            confidence=0.0,
        )

    return QueryRoute(
        intent=data.get("intent", "general"),
        rewritten_query=data.get("rewritten_query", question),
        document_id=data.get("document_id"),
        jurisdiction=data.get("jurisdiction"),
        authority=data.get("authority"),
        confidence=float(data.get("confidence", 0.0)),
    )