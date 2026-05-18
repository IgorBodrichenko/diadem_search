import os
from collections import Counter
from pinecone import Pinecone

api_key = os.environ.get("PINECONE_API_KEY")
index_name = os.environ.get("PINECONE_INDEX_NAME")

if not api_key or not index_name:
    raise RuntimeError("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME")

pc = Pinecone(api_key=api_key)
idx = pc.Index(index_name)

stats = idx.describe_index_stats()
print("TOTAL_VECTORS", stats.get("total_vector_count"))

all_ids = []
for batch in idx.list(namespace=""):
    if isinstance(batch, list):
        all_ids.extend(batch)
    elif isinstance(batch, dict):
        vectors = batch.get("vectors") or []
        all_ids.extend(v.get("id") for v in vectors if isinstance(v, dict) and v.get("id"))

print("IDS_COLLECTED", len(all_ids))

file_names = Counter()
doc_ids = Counter()
concepts = Counter()
errors = 0

for vid in all_ids:
    try:
        result = idx.query(id=vid, top_k=1, include_metadata=True, namespace="")
        matches = result.get("matches") or []
        if not matches:
            continue

        md = (matches[0].get("metadata") or {})
        if md.get("file_name"):
            file_names[md["file_name"]] += 1
        if md.get("doc_id"):
            doc_ids[md["doc_id"]] += 1
        if md.get("concept"):
            concepts[md["concept"]] += 1
    except Exception:
        errors += 1

print("UNIQUE_FILES", len(file_names))
for name, cnt in sorted(file_names.items()):
    print(f"FILE\t{name}\t{cnt}")

print("UNIQUE_DOC_IDS", len(doc_ids))
print("QUERY_ERRORS", errors)
print("TOP_CONCEPTS")
for name, cnt in concepts.most_common(20):
    print(f"CONCEPT\t{name}\t{cnt}")
