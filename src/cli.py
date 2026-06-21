import argparse
from pathlib import Path

from config import RAGConfig
from pipeline import build_index, answer_question


def print_sources(sources):
    print()
    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

    for source in sources:
        print()
        print(f"[Source {source.source_id}]")
        print(f"Document: {source.document_name}")
        print(f"Authority: {source.authority}")
        print(f"Status: {source.status}")
        print(f"Document ID: {source.document_id}")
        print(f"Segment: {source.segment_position}")
        print(f"Distance: {source.distance}")
        print(f"URL: {source.source_url}")


def main():
    parser = argparse.ArgumentParser(
        description="RAG system for AI governance documents."
    )

    parser.add_argument(
        "--build",
        action="store_true",
        help="Build or rebuild the ChromaDB index.",
    )

    parser.add_argument(
        "--ask",
        type=str,
        help="Ask a question over the indexed documents.",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing documents.csv, segments.csv, authorities.csv, and collections.csv.",
    )

    parser.add_argument(
        "--db-dir",
        type=Path,
        default=Path("chroma_db"),
        help="Directory where ChromaDB index is stored.",
    )

    parser.add_argument(
        "--n-results",
        type=int,
        default=8,
        help="Number of chunks to retrieve.",
    )

    parser.add_argument(
        "--embed-model",
        type=str,
        default="nomic-embed-text:latest",
        help="Ollama embedding model.",
    )

    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama3.1:8b",
        help="Ollama chat model.",
    )

    args = parser.parse_args()

    config = RAGConfig(
        data_dir=args.data_dir,
        db_dir=args.db_dir,
        embed_model=args.embed_model,
        llm_model=args.llm_model,
        default_n_results=args.n_results,
    )

    if args.build:
        build_index(config)

    if args.ask:
        result = answer_question(
            question=args.ask,
            config=config,
            n_results=args.n_results,
        )

        print()
        print("=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(result.answer)

        print_sources(result.sources)

    if not args.build and not args.ask:
        parser.print_help()