import pandas as pd


TAX_PREFIXES = [
    "Applications:",
    "Harms:",
    "Incentives:",
    "Risk factors:",
    "Strategies:",
]


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


def get_jurisdiction(authority_name, auth_lookup):
    if pd.isna(authority_name):
        return ""

    if authority_name not in auth_lookup.index:
        return ""

    jurisdiction = auth_lookup.loc[authority_name, "Jurisdiction"]

    if pd.isna(jurisdiction):
        return ""

    return str(jurisdiction)


def create_metadata(documents, authorities):
    auth_lookup = authorities.set_index("Name")
    tax_cols = [
        c for c in documents.columns
        if any(c.startswith(p) for p in TAX_PREFIXES)
    ]

    meta = {}

    for row_index, row in documents.iterrows():
        doc_id = to_int_id(row["AGORA ID"], "AGORA ID")

        if doc_id is None:
            continue

        active_tags = [c for c in tax_cols if to_bool(row[c])]

        collections = (
            [c.strip() for c in str(row["Collections"]).split(";") if c.strip()]
            if pd.notna(row["Collections"])
            else []
        )

        authority = row["Authority"]
        jurisdiction = get_jurisdiction(authority, auth_lookup)

        official_name = row["Official name"]
        casual_name = (
            row["Casual name"]
            if pd.notna(row["Casual name"])
            else official_name
        )

        meta[doc_id] = {
            "document_id": doc_id,
            "official_name": official_name,
            "casual_name": casual_name,
            "authority": authority,
            "jurisdiction": jurisdiction,
            "collections": collections,
            "status": row["Most recent activity"],
            "status_date": row["Most recent activity date"],
            "short_summary": row["Short summary"] if pd.notna(row["Short summary"]) else "",
            "long_summary": row["Long summary"] if pd.notna(row["Long summary"]) else "",
            "source_url": row["Link to document"] if pd.notna(row["Link to document"]) else "",
            "annotated": to_bool(row["Annotated?"]),
            "validated": to_bool(row["Validated?"]),
            "active_tags": active_tags,
        }

    return meta

def find_matching_document_ids(query, meta, min_match_len=6):
    query_lower = query.lower()
    matches = {}

    for doc_id, doc_meta in meta.items():
        names = [
            str(doc_meta.get("official_name", "")),
            str(doc_meta.get("casual_name", "")),
        ]

        for name in names:
            name_lower = name.lower().strip()

            if not name_lower or name_lower == "nan" or len(name_lower) < min_match_len:
                continue

            if name_lower in query_lower:
                match_len = len(name_lower)
                if match_len > matches.get(doc_id, 0):
                    matches[doc_id] = match_len

    ranked = sorted(matches.items(), key=lambda pair: pair[1], reverse=True)
    return [int(doc_id) for doc_id, _ in ranked]


