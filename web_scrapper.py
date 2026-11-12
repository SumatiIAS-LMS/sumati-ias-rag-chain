import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlparse, urlunparse

os.environ.setdefault("USER_AGENT", "Mozilla/5.0")

from bson import ObjectId
from langchain_community.document_loaders import WebBaseLoader
from langchain_mongodb.index import create_vector_search_index
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import ASCENDING

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SUMATI_IAS_DATA_CHUNKS_VECTOR_INDEX_NAME,
    EMBEDDING_DIMENSIONS,
)
from rag_chain import get_embeddings

_indexes_ready = False


def clean(text: str) -> str:
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def safe_name(url: str) -> str:
    """Build a file-friendly name from the URL, e.g. example_com_blog_post."""
    p = urlparse(url)
    raw = f"{p.netloc}{p.path}".strip("/")
    return re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_")[:80] or "page"


def fetch_page(url: str) -> dict:
    """Download one public page and return its title and cleaned text."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an http or https link")

    loader = WebBaseLoader(
        web_path=url,
        header_template={"User-Agent": "Mozilla/5.0"},
        requests_kwargs={"timeout": 20},
    )
    docs = loader.load()
    if not docs:
        raise ValueError("page returned no content")

    parts = []
    title = None
    for doc in docs:
        if not title:
            title = doc.metadata.get("title")
        if doc.page_content and doc.page_content.strip():
            parts.append(clean(doc.page_content))

    return {
        "url": url,
        "title": title,
        "content": clean("\n\n".join(parts)),
    }


def split_into_chunks(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return [
        chunk.strip()
        for chunk in splitter.split_text(text)
        if chunk and chunk.strip()
    ]


def _as_vector(values) -> list[float]:
    return [float(value) for value in values]


def ensure_indexes(chunks_collection) -> None:
    global _indexes_ready
    if _indexes_ready:
        return

    chunks_collection.create_index(
        [("documentId", ASCENDING), ("resourceType", ASCENDING)],
        name="document_resource",
    )
    _ensure_vector_index(chunks_collection)
    _indexes_ready = True


def _ensure_vector_index(chunks_collection) -> None:
    """Create the Atlas vector index when this cluster supports search indexes."""
    try:
        existing = {
            index.get("name")
            for index in chunks_collection.list_search_indexes()
        }
        if SUMATI_IAS_DATA_CHUNKS_VECTOR_INDEX_NAME in existing:
            return

        create_vector_search_index(
            collection=chunks_collection,
            index_name=SUMATI_IAS_DATA_CHUNKS_VECTOR_INDEX_NAME,
            dimensions=EMBEDDING_DIMENSIONS,
            path="embedding",
            similarity="cosine",
            filters=["documentId", "resourceType", "sourceCollection"],
        )
    except Exception as exc:
        print(f"[WARN] vector search index was not created: {exc}")


def ingest_web_pages(pages: list[dict], chunks_collection) -> dict:
    """Fetch each page, split it, embed the chunks, and store one record per chunk.

    Chunks already stored for the same document and resource type are removed
    only after the new chunks have been inserted.
    """
    ensure_indexes(chunks_collection)

    results = []
    inserted_total = 0

    for page in pages:
        url = (page.get("url") or "").strip()
        document_id = (page.get("documentId") or "").strip()
        resource_type = (page.get("resourceType") or "").strip()
        result = {
            "documentId": document_id,
            "sourceUrl": url,
            "title": page.get("title") or "",
            "slug": page.get("slug") or "",
        }

        if not url or not document_id or not resource_type:
            result["error"] = "url, documentId, and resourceType are required"
            results.append(result)
            continue

        if not ObjectId.is_valid(document_id):
            result["error"] = "documentId is not a valid id"
            results.append(result)
            continue

        try:
            fetched = fetch_page(url)
        except Exception as exc:
            print(f"[FAILED] {url} -> {exc}")
            result["error"] = f"could not fetch page: {exc}"
            results.append(result)
            continue

        chunks = split_into_chunks(fetched["content"])
        if not chunks:
            print(f"[FAILED] {url} -> no extractable text")
            result["error"] = "page has no extractable text"
            results.append(result)
            continue

        now = datetime.now(timezone.utc)
        document_oid = ObjectId(document_id)
        try:
            vectors = get_embeddings().embed_documents(chunks)
            if len(vectors) != len(chunks):
                raise ValueError("embedding count did not match the chunks")

            records = [
                {
                    "documentId": document_oid,
                    "resourceType": resource_type,
                    "sourceCollection": page.get("sourceCollection") or "",
                    "sourceUrl": url,
                    "title": page.get("title") or "",
                    "slug": page.get("slug") or "",
                    "pageTitle": fetched.get("title") or "",
                    "chunkIndex": index,
                    "text": chunk,
                    "embedding": _as_vector(vectors[index]),
                    "createdAt": now,
                }
                for index, chunk in enumerate(chunks)
            ]
            inserted = chunks_collection.insert_many(records)
            chunks_collection.delete_many({
                "documentId": document_oid,
                "resourceType": resource_type,
                "_id": {"$nin": inserted.inserted_ids},
            })
        except Exception as exc:
            print(f"[FAILED] {url} -> {exc}")
            result["error"] = f"could not store chunks: {exc}"
            results.append(result)
            continue

        inserted_count = len(inserted.inserted_ids)
        inserted_total += inserted_count
        result["chunks"] = inserted_count
        result["inserted"] = inserted_count
        results.append(result)
        print(f"[OK] {url} -> {inserted_count} chunks")

    return {
        "inserted": inserted_total,
        "results": results,
    }


def ingest_extracted_documents(documents: list[dict], chunks_collection) -> dict:
    """Chunk, embed, and store text using the same records as scraped pages.

    Chunks already stored for the same document, resource type, and source
    are removed only after the new chunks have been inserted.
    """
    ensure_indexes(chunks_collection)

    results = []
    inserted_total = 0

    for document in documents:
        source_url = (document.get("sourceUrl") or "").strip()
        document_id = (document.get("documentId") or "").strip()
        resource_type = (document.get("resourceType") or "").strip()
        text = document.get("text") or ""
        result = {
            "documentId": document_id,
            "sourceUrl": source_url,
            "title": document.get("title") or "",
            "slug": document.get("slug") or "",
        }

        if not source_url or not document_id or not resource_type:
            result["error"] = "sourceUrl, documentId, and resourceType are required"
            results.append(result)
            continue

        if not ObjectId.is_valid(document_id):
            result["error"] = "documentId is not a valid id"
            results.append(result)
            continue

        chunks = split_into_chunks(text)
        if not chunks:
            result["error"] = "document has no extractable text"
            results.append(result)
            continue

        now = datetime.now(timezone.utc)
        document_oid = ObjectId(document_id)
        try:
            vectors = get_embeddings().embed_documents(chunks)
            if len(vectors) != len(chunks):
                raise ValueError("embedding count did not match the chunks")

            records = [
                {
                    "documentId": document_oid,
                    "resourceType": resource_type,
                    "sourceCollection": document.get("sourceCollection") or "",
                    "sourceUrl": source_url,
                    "title": document.get("title") or "",
                    "slug": document.get("slug") or "",
                    "pageTitle": document.get("pageTitle") or "",
                    "chunkIndex": index,
                    "text": chunk,
                    "embedding": _as_vector(vectors[index]),
                    "createdAt": now,
                }
                for index, chunk in enumerate(chunks)
            ]
            inserted = chunks_collection.insert_many(records)
            chunks_collection.delete_many({
                "documentId": document_oid,
                "resourceType": resource_type,
                "sourceUrl": source_url,
                "_id": {"$nin": inserted.inserted_ids},
            })
        except Exception as exc:
            print(f"[FAILED] {source_url} -> {exc}")
            result["error"] = f"could not store chunks: {exc}"
            results.append(result)
            continue

        inserted_count = len(inserted.inserted_ids)
        inserted_total += inserted_count
        result["pages"] = document.get("pages") or 0
        result["chunks"] = inserted_count
        result["inserted"] = inserted_count
        results.append(result)
        print(f"[OK] {source_url} -> {inserted_count} chunks")

    return {
        "inserted": inserted_total,
        "results": results,
    }


def extract_urls(urls: list[str], output_dir: str = "output") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_records = []

    for url in urls:
        try:
            record = fetch_page(url)
        except Exception as e:
            print(f"[FAILED] {url} -> {e}")
            continue

        all_records.append(record)

        txt_path = out / f"{safe_name(url)}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"URL: {url}\n")
            f.write(f"Title: {record['title']}\n")
            f.write("-" * 60 + "\n")
            f.write(record["content"])

        print(f"[OK] {url} -> {txt_path} ({len(record['content'])} chars)")

    json_path = out / "all_pages.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    combined_path = out / "all_pages.txt"
    with open(combined_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(f"URL: {r['url']}\n")
            f.write(f"Title: {r['title']}\n")
            f.write("-" * 60 + "\n")
            f.write(r["content"] + "\n\n" + "=" * 60 + "\n\n")

    print(f"\nDone. {len(all_records)}/{len(urls)} pages saved in '{out}/'")


if __name__ == "__main__":
    urls = [
        # "https://srv2064600.hstgr.cloud",
        # "https://srv2064600.hstgr.cloud/daily-current-affairs",
        # "https://srv2064600.hstgr.cloud/web-hot-topics",
        # "https://srv2064600.hstgr.cloud/web-editorial",
        # "https://srv2064600.hstgr.cloud/monthly-current-affairs",
        # "https://srv2064600.hstgr.cloud/upsc-article",
        "https://srv2064600.hstgr.cloud/daily-current-affairs/31-october-2026",
        # "https://www.sumatiias.com/newsanalysis-details?date=2026-11-05"
    ]
    extract_urls(urls)
