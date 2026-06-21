import re
import pandas as pd

def to_int_id(value, column_name):
    if pd.isna(value):
        return None

    try:
        return int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {column_name}: {value!r}")


def to_bool(value):
    if pd.isna(value):
        return False

    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y"}

    return bool(value)
def create_embedding_text(doc_meta, segment_summary, raw_text):
    header_bits = [doc_meta["official_name"]]

    if doc_meta.get("casual_name"):
        header_bits.append(f"Also known as {doc_meta['casual_name']}")

    if doc_meta.get("authority"):
        header_bits.append(f"Issued by {doc_meta['authority']}")

    if doc_meta.get("status"):
        header_bits.append(
            f"Status: {doc_meta['status']} ({doc_meta['status_date']})"
        )

    parts = [" | ".join(header_bits)]

    if segment_summary:
        parts.append(f"Summary: {segment_summary}")

    parts.append(raw_text)

    return "\n\n".join(parts)


def split_long_text(text, max_words, overlap):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        if current_len + word_count > max_words and current:
            chunks.append(" ".join(current))

            overlap_sentences = []
            overlap_len = 0

            for previous_sentence in reversed(current):
                overlap_sentences.insert(0, previous_sentence)
                overlap_len += len(previous_sentence.split())

                if overlap_len >= overlap:
                    break

            current = overlap_sentences
            current_len = overlap_len

        current.append(sentence)
        current_len += word_count

    if current:
        chunks.append(" ".join(current))

    return chunks


def create_chunks(segments, meta, config):
    chunks = []

    for _, row in segments.iterrows():
        doc_id = to_int_id(row["Document ID"], "Document ID")

        if doc_id is None or doc_id not in meta:
            continue

        if "Text" not in row or row["Text"] != row["Text"]:
            continue

        raw_text = str(row["Text"]).strip()

        if not raw_text:
            continue

        doc_meta = meta[doc_id]

        segment_summary = ""
        if "Summary" in row and row["Summary"] == row["Summary"]:
            segment_summary = str(row["Summary"]).strip()

        segment_position = to_int_id(row["Segment position"], "Segment position")

        if segment_position is None:
            continue

        word_count = len(raw_text.split())

        if word_count > config.split_word_threshold:
            sub_texts = split_long_text(
                raw_text,
                max_words=config.subchunk_words,
                overlap=config.subchunk_overlap_words,
            )
        else:
            sub_texts = [raw_text]

        for part_index, sub_text in enumerate(sub_texts):
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_{segment_position}_p{part_index}",
                    "document_id": doc_id,
                    "segment_position": segment_position,
                    "part_index": part_index,
                    "text": sub_text,
                    "embedding_text": create_embedding_text(
                        doc_meta,
                        segment_summary,
                        sub_text,
                    ),
                    "word_count": len(sub_text.split()),
                    "segment_summary": segment_summary,
                    "non_operative": to_bool(row["Non-operative"]),
                    "not_ai_related": to_bool(row["Not AI-related"]),
                    "document": doc_meta,
                }
            )

    return chunks