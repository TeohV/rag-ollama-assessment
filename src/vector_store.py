import chromadb

from ollama_client import get_embedding


def get_collection(config):
    client = chromadb.PersistentClient(path=str(config.db_dir))
    return client.get_or_create_collection(name=config.collection_name)


def reset_collection(config):
    client = chromadb.PersistentClient(path=str(config.db_dir))

    try:
        client.delete_collection(name=config.collection_name)
        print("Old collection deleted.")
    except Exception:
        print("No old collection to delete.")


def chunk_to_chroma_record(chunk):
    document = chunk["document"]

    metadata = {
        "document_id": int(chunk["document_id"]),
        "segment_position": int(chunk["segment_position"]),
        "word_count": int(chunk["word_count"]),
        "segment_summary": str(chunk["segment_summary"]),
        "non_operative": bool(chunk["non_operative"]),
        "not_ai_related": bool(chunk["not_ai_related"]),

        "official_name": str(document["official_name"]),
        "casual_name": str(document["casual_name"]),
        "authority": str(document["authority"]),
        "jurisdiction": str(document["jurisdiction"] or ""),
        "status": str(document["status"]),
        "status_date": str(document["status_date"]),
        "source_url": str(document["source_url"]),

        "collections": "; ".join(document["collections"]),
        "active_tags": "; ".join(document["active_tags"]),
        "annotated": bool(document["annotated"]),
        "validated": bool(document["validated"]),
    }

    return chunk["chunk_id"], chunk["text"], metadata


def store_chunks(chunks, config):
    collection = get_collection(config)

    for chunk in chunks:
        chunk_id, document_text, metadata = chunk_to_chroma_record(chunk)
        embedding = get_embedding(chunk["embedding_text"], config)

        collection.upsert(
            ids=[chunk_id],
            documents=[document_text],
            metadatas=[metadata],
            embeddings=[embedding],
        )

        print("Stored chunk:", chunk_id)


def query_collection(query, config, n_results=8, document_id=None):
    collection = get_collection(config)
    query_embedding = get_embedding(query, config)

    filters = [
        {"non_operative": False},
        {"not_ai_related": False},
    ]

    if document_id is not None:
        filters.append({"document_id": int(document_id)})

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"$and": filters},
    )