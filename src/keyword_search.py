import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def _index_path(config):
    return Path(config.db_dir) / "keyword_index.pkl"


def build_keyword_index(chunks, config):
    if not chunks:
        return

    chunk_ids = []
    tokenized_corpus = []
    records = {}

    for chunk in chunks:
        document = chunk["document"]

        chunk_ids.append(chunk["chunk_id"])
        tokenized_corpus.append(tokenize(chunk["text"]))

        records[chunk["chunk_id"]] = {
            "text": chunk["text"],
            "document_id": chunk["document_id"],
            "segment_position": chunk["segment_position"],
            "official_name": document["official_name"],
            "authority": document["authority"],
            "status": document["status"],
            "source_url": document["source_url"],
            "non_operative": chunk["non_operative"],
            "not_ai_related": chunk["not_ai_related"],
        }

    bm25 = BM25Okapi(tokenized_corpus)

    with open(_index_path(config), "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids, "records": records}, f)

    print("Keyword index built:", len(chunk_ids), "chunks")


_CACHE = {}


def _load_index(config):
    path = _index_path(config)
    cache_key = str(path)

    if cache_key not in _CACHE:
        if not path.exists():
            return None

        with open(path, "rb") as f:
            _CACHE[cache_key] = pickle.load(f)

    return _CACHE[cache_key]


def keyword_search(query, config, n_results=8, document_id=None):
    index = _load_index(config)

    if index is None:
        return []

    tokens = tokenize(query)

    if not tokens:
        return []

    scores = index["bm25"].get_scores(tokens)

    ranked = sorted(
        zip(index["chunk_ids"], scores),
        key=lambda pair: pair[1],
        reverse=True,
    )

    results = []

    for chunk_id, score in ranked:
        if score <= 0:
            break

        record = index["records"][chunk_id]

        if record["non_operative"] or record["not_ai_related"]:
            continue

        if document_id is not None and record["document_id"] != int(document_id):
            continue

        results.append((chunk_id, record))

        if len(results) >= n_results:
            break

    return results
