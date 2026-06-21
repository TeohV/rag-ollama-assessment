from dataclasses import dataclass, replace

from vector_store import query_collection
from keyword_search import keyword_search


@dataclass
class RetrievedChunk:
    source_id: int
    text: str
    document_name: str
    authority: str
    status: str
    source_url: str
    document_id: int
    segment_position: int
    distance: float | None


def _vector_candidates(query, config, n_candidates, document_id):
    results = query_collection(
        query=query,
        config=config,
        n_results=n_candidates,
        document_id=document_id,
    )

    if not results["ids"] or not results["ids"][0]:
        return [], {}

    ids = results["ids"][0]
    lookup = {
        chunk_id: {"text": text, "metadata": metadata, "distance": distance}
        for chunk_id, text, metadata, distance in zip(
            ids,
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    }
    return ids, lookup


def retrieve_chunks(query, config, n_results=8, document_id=None, n_candidates=30):
    vector_ids, vector_lookup = _vector_candidates(
        query, config, n_candidates, document_id
    )
    keyword_results = keyword_search(
        query, config, n_results=n_candidates, document_id=document_id
    )
    keyword_lookup = {chunk_id: record for chunk_id, record in keyword_results}

    RRF_K = 60
    scores = {}

    for rank, chunk_id in enumerate(vector_ids):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

    for rank, chunk_id in enumerate(keyword_lookup):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

    if not scores:
        return []

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:n_results]

    retrieved = []

    for source_id, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in vector_lookup:
            entry = vector_lookup[chunk_id]
            metadata, text, distance = entry["metadata"], entry["text"], entry["distance"]
        else:
            record = keyword_lookup[chunk_id]
            metadata, text, distance = record, record["text"], None

        retrieved.append(
            RetrievedChunk(
                source_id=source_id,
                text=text,
                document_name=metadata["official_name"],
                authority=metadata["authority"],
                status=metadata["status"],
                source_url=metadata["source_url"],
                document_id=int(metadata["document_id"]),
                segment_position=int(metadata["segment_position"]),
                distance=distance,
            )
        )

    return retrieved


def _split_quota(n_results, n_groups):
    base = n_results // n_groups
    remainder = n_results % n_groups
    return [base + (1 if i < remainder else 0) for i in range(n_groups)]


def retrieve_chunks_for_documents(query, config, document_ids, n_results=8, n_candidates=30):
    if not document_ids:
        return retrieve_chunks(query, config, n_results=n_results, n_candidates=n_candidates)

    quotas = _split_quota(n_results, len(document_ids))

    all_chunks = []
    for doc_id, quota in zip(document_ids, quotas):
        if quota <= 0:
            continue

        chunks = retrieve_chunks(
            query,
            config,
            n_results=quota,
            document_id=doc_id,
            n_candidates=n_candidates,
        )
        all_chunks.extend(chunks)
    return [
        replace(chunk, source_id=source_id)
        for source_id, chunk in enumerate(all_chunks, start=1)
    ]
