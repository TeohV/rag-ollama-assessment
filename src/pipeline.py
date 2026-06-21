from dataclasses import dataclass

from config import RAGConfig
from data_loader import load_data
from metadata import create_metadata, find_matching_document_ids
from chunking import create_chunks
from vector_store import reset_collection, store_chunks
from keyword_search import build_keyword_index
from retrieval import retrieve_chunks, retrieve_chunks_for_documents, RetrievedChunk
from generation import generate_answer
from query_router import classify_query


@dataclass
class AnswerResult:
    answer: str
    sources: list[RetrievedChunk]


def build_index(config: RAGConfig):
    documents, segments, authorities, _ = load_data(config.data_dir)

    metadata = create_metadata(documents, authorities)
    chunks = create_chunks(segments, metadata, config)

    print("Metadata records:", len(metadata))
    print("Chunks created:", len(chunks))

    reset_collection(config)
    store_chunks(chunks, config)
    build_keyword_index(chunks, config)

    print()
    print("Indexing complete.")


def answer_question(question, config: RAGConfig, n_results=None):
    documents, _, authorities, _ = load_data(config.data_dir)
    metadata = create_metadata(documents, authorities)

    matched_document_ids = find_matching_document_ids(question, metadata)

    # The user's wording doesn't always match the source documents' wording
    # (e.g. "refund" vs. "reimbursement"). Ask the LLM to rephrase the
    # question for retrieval before embedding it. Best-effort: if this call
    # fails for any reason (Ollama down, bad JSON, etc.) we fall back to the
    # raw question rather than failing the whole request.
    retrieval_query = question
    try:
        route = classify_query(question, metadata, config)
        if route.rewritten_query:
            retrieval_query = route.rewritten_query
    except Exception:
        pass

    n = n_results or config.default_n_results

    if len(matched_document_ids) >= 2:
        # Comparison-style query naming multiple documents: guarantee each
        # named document gets retrieval slots instead of one pooled search
        # letting the higher-scoring document crowd out the other.
        sources = retrieve_chunks_for_documents(
            query=retrieval_query,
            config=config,
            document_ids=matched_document_ids,
            n_results=n,
        )
    else:
        document_id = matched_document_ids[0] if matched_document_ids else None
        sources = retrieve_chunks(
            query=retrieval_query,
            config=config,
            n_results=n,
            document_id=document_id,
        )

    if not sources:
        return AnswerResult(
            answer="I could not find relevant sources in the indexed corpus.",
            sources=[],
        )

    answer = generate_answer(question, sources, config)

    return AnswerResult(answer=answer, sources=sources)