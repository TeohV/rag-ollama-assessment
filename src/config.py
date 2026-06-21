from dataclasses import dataclass
from pathlib import Path


@dataclass
class RAGConfig:
    data_dir: Path = Path("data")
    db_dir: Path = Path("chroma_db")
    collection_name: str = "agora_ai_governance"

    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text:latest"
    llm_model: str = "llama3.1:8b"

    split_word_threshold: int = 500
    subchunk_words: int = 350
    subchunk_overlap_words: int = 50
    default_n_results: int = 8
