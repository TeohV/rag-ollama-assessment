from pathlib import Path
import pandas as pd


def load_data(data_dir: Path):
    required_files = {
        "documents": data_dir / "documents.csv",
        "segments": data_dir / "segments.csv",
        "authorities": data_dir / "authorities.csv",
        "collections": data_dir / "collections.csv",
    }

    missing = [str(path) for path in required_files.values() if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required data files:\n"
            + "\n".join(missing)
            + "\n\nPlace the CSV files inside the data/ directory."
        )

    documents = pd.read_csv(required_files["documents"])
    segments = pd.read_csv(required_files["segments"])
    authorities = pd.read_csv(required_files["authorities"])
    collections_df = pd.read_csv(required_files["collections"])

    return documents, segments, authorities, collections_df