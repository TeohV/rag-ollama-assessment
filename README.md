# RAG System for AI Governance Documents

## How It Works

1. **Load & join metadata** (`data_loader.py`, `metadata.py`) — reads
   `documents.csv`, `segments.csv`, `authorities.csv`, `collections.csv` and
   joins each document to its issuing authority's jurisdiction.
2. **Chunk** (`chunking.py`) — each segment becomes one chunk. Segments over
   500 words are split into ~350-word overlapping sub-chunks (50-word overlap)
   on sentence boundaries, so no chunk loses mid-sentence context. Every
   chunk's embedding text is prefixed with document metadata (official name,
   issuing authority, status) so retrieval can match on "what is this
   document" as well as "what does this passage say."
3. **Index** (`vector_store.py`, `keyword_search.py`) — chunks are embedded
   with `nomic-embed-text` and stored in ChromaDB, and a parallel BM25 index
   is built over the raw chunk text.
4. **Retrieve** (`retrieval.py`) — queries run against both indexes and are
   fused with Reciprocal Rank Fusion (RRF). Dense embeddings alone tend to
   miss exact terms (section numbers, named entities); BM25 alone misses
   paraphrases. RRF means a chunk only has to rank well on *either* signal.
   If a query names two or more documents explicitly, each gets a guaranteed
   slot quota instead of one pooled search letting a higher-scoring document
   crowd the other out (see Limitations).
5. **Generate** (`generation.py`, `query_router.py`) — the question is
   optionally rewritten by the LLM for retrieval (best-effort; falls back to
   the raw question on failure), sources are formatted into a grounding
   prompt instructing the model to cite `[Source N]` inline and never invent
   article/section numbers, and `llama3.1:8b` generates the answer.
6. **Verify** (`generation.py: _verify_citations`) — a lightweight
   proximity-based check flags any `Article/Section/Sec./§/Clause/Annex N`
   cited near a `[Source N]` tag that doesn't actually appear in that
   source's retrieved text, and appends a warning if so. Best-effort, not a
   full citation auditor (see Limitations).

## Key Decisions

- **Hybrid retrieval (vector + BM25) over vector-only.** Legal/regulatory
  text leans heavily on exact terms — "Article 20," specific named entities,
  section numbers — that don't always embed close to how a user phrases a
  question. RRF widens recall without needing query-specific tuning.
- **Per-document quota retrieval for comparison queries.** A single pooled
  search for "compare X and Y" can let one document's chunks dominate every
  result slot, leaving the model no choice but to answer the other half from
  memory. Detecting named documents and giving each a guaranteed slice fixes
  this for the case it was built for (see Limitations for where it doesn't
  yet trigger).
- **Citation verification as a safety net, not a guarantee.** The prompt
  instructs the model not to invent section numbers; the proximity check
  catches the common failure mode (a plausible-sounding number that isn't in
  the retrieved text) cheaply, without claiming to catch everything.
- **Excluding non-operative / non-AI-related segments at both the vector and
  keyword layer.** Some segments in the corpus are preambles, definitions
  carried over from unrelated law, or otherwise not substantive obligations;
  both retrieval paths filter these out so they can't surface as answers.
- **Sentence-boundary sub-chunking with overlap**, rather than splitting
  long segments at a fixed word count, so a chunk boundary never lands
  mid-sentence and lose a key clause.

## Limitations / Known Issues

- **No relevance threshold on retrieval.** Out-of-domain or unanswerable
  questions still return the top-N chunks by RRF score, even when none are
  actually relevant (distances ~0.72+ vs. ~0.5–0.6 for genuinely relevant
  matches). The model usually catches this in its answer text, but it can
  still drift into hedged, ungrounded "general commercial practice" filler
  rather than cleanly stating no relevant source exists. A distance/score
  cutoff that short-circuits to "I could not find relevant sources" (the
  pipeline already has this path for zero-result cases) would close this.
- **Multi-document retrieval guarantee only triggers on exact name
  matching.** `find_matching_document_ids` matches the query against each
  document's official/casual name as a substring. A query that *describes*
  two frameworks without naming either precisely enough (e.g. asking about
  "NIST's update timeframe" and "EU AI Act fine thresholds" in one
  question) falls back to a single pooled search, where we observed one
  document's chunks filling all 8 result slots and the other being entirely
  absent from the answer. `query_router.py` already classifies query intent
  and could supply document candidates to supplement the substring matcher,
  but isn't currently wired into that decision.
- **Citation verification regex is narrow.** It only catches
  `Article/Section/Sec./§/Clause/Annex N` patterns. Framework-specific
  enumerators — e.g. NIST RMF's "MANAGE 1.2" — aren't checked, which is
  exactly where we saw the model cite plausible-but-unverified sub-numbers.
- **19 segments in `segments.csv` reference a Document ID with no matching
  row in `documents.csv`**, and 4 documents have zero segments associated
  with them. Both are silently skipped during chunking (no crash), but
  surfaced here for transparency rather than left undocumented.
- **Indexing is sequential, not batched.** `store_chunks` embeds one chunk
  per HTTP call to Ollama with no batching or resume-on-failure. On the full
  corpus (~5,300 base segments before sub-splitting) a full `--build` run
  takes a while and would need to restart from scratch if interrupted.
- **The 8B local model occasionally drifts toward outside knowledge** on
  questions with no good source match, despite the prompt instructing it to
  use only provided sources. The citation-window check doesn't catch this
  pattern of hedged generic filler since it isn't attached to a `[Source N]`
  tag.

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull required Ollama models
ollama pull nomic-embed-text
ollama pull llama3.1:8b

# 3. Place documents.csv, segments.csv, authorities.csv, collections.csv
#    in data/ (download from thttps://www.kaggle.com/datasets/umerhaddii/ai-governance-documents-data)

# 4. Build the index (one-time, or after data changes)
python app.py --build

# 5. Ask a question
python app.py --ask "What obligations apply to providers of high-risk AI systems?"
```

