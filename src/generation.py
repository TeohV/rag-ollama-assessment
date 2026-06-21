from ollama_client import chat
import re


_PROVISION_RE = re.compile(
    r"\b(?:Article|Section|Sec\.|§|Clause|Annex)\s+\d+[A-Za-z]?\b", re.IGNORECASE
)
_SOURCE_TAG_RE = re.compile(r"\[Source\s+(\d+)[^\]]*\]")


def build_answer_prompt(question, sources):
    context = "\n\n".join(
        f"""[Source {source.source_id}]
Document: {source.document_name}
Authority: {source.authority}
Status: {source.status}
Document ID: {source.document_id}
Segment: {source.segment_position}
URL: {source.source_url}

{source.text}
"""
        for source in sources
    )

    return f"""
You are answering questions about AI governance documents.

Use only the provided sources. Do not use outside knowledge.

Question:
{question}

Instructions:
1. Answer the question directly.
2. Focus on legal or policy requirements, obligations, duties, conditions, or compliance steps.
3. Cite sources inline using [Source N].
4. Only cite a specific article, section, or clause number (e.g. "Article 20") if that exact number appears in the source text below. If a source doesn't show one, refer to it by document name only -- never infer or guess a number.
5. Mention the document name for each major point.
6. If the sources are incomplete, say what information is missing.
7. Do not invent requirements that are not supported by the sources.

Sources:
{context}

Answer:
"""


def _verify_citations(answer, sources, window=200):
    """Best-effort check: flag specific article/section numbers cited near a
    [Source N] tag that don't actually appear in that source's retrieved
    text. The prompt instructs the model not to invent these, but LLMs
    still sometimes produce a plausible-sounding number instead of reading
    it from the chunk -- this catches the common case cheaply. Proximity
    based, not perfect, but a useful safety net rather than trusting the
    model's citation discipline alone.
    """
    sources_by_id = {s.source_id: s for s in sources}
    warnings = []
    seen = set()

    for match in _SOURCE_TAG_RE.finditer(answer):
        source_id = int(match.group(1))
        source = sources_by_id.get(source_id)

        if source is None:
            continue

        start = max(0, match.start() - window)
        end = min(len(answer), match.end() + window)
        window_text = answer[start:end]

        for provision_match in _PROVISION_RE.finditer(window_text):
            provision = provision_match.group(0)
            key = (source_id, provision.lower())

            if key in seen:
                continue
            seen.add(key)

            if provision.lower() not in source.text.lower():
                warnings.append(
                    f'"{provision}" cited near [Source {source_id}] but not found '
                    f"verbatim in that excerpt's text"
                )

    return warnings


def generate_answer(question, sources, config):
    prompt = build_answer_prompt(question, sources)
    answer = chat(prompt, config)

    warnings = _verify_citations(answer, sources)
    if warnings:
        answer += (
            "\n\n---\n"
            "⚠️ Unverified citations (not found verbatim in the retrieved text -- "
            "double-check against the source before relying on these):\n"
        )
        answer += "\n".join(f"- {w}" for w in warnings)

    return answer