import os
import tempfile

from bson import ObjectId
from flask import Blueprint, current_app, jsonify
from pypdf import PdfReader

from config import MONGODB_DATA_CHUNKS_COLLECTION
from rag_chain import database
from server import UPLOAD_DIR, download_pdf, request_data, resolve_stored_file
from web_scrapper import ingest_extracted_documents

public_pdf_bp = Blueprint("pdf", __name__)

COLLECTION_FILES = {
    "strikers": {
        "title_field": "title",
        "url_fields": ["pdfFile"],
    },
    "indiaplacenews": {
        "title_field": "title",
        "url_fields": ["pdfFile"],
    },
    "webresources": {
        "title_field": "title",
        "url_fields": ["pdfUrl"],
    },
    "ncertbooks": {
        "title_field": "bookTitle",
        "url_fields": ["bookPdf", "bookUrl"],
    },
    "magazines": {
        "title_field": "title",
        "url_fields": ["pdfFile"],
    }
}

FILE_LABELS = {
    "questionPaperPdf": "Question Paper",
    "answerKeyPdf": "Answer Key",
}


def _safe_error(exc):
    message = str(exc).split("?", 1)[0].strip() or "could not read pdf"
    if "X-Amz-" in message:
        return "could not download pdf"
    return message[:300]


def _text(value):
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _document_title(document, title_field):
    title = document.get(title_field) or document.get("title") or ""
    if not isinstance(title, str):
        title = str(title)
    return title.strip()


def _stored_files(document, url_fields):
    public_url = _text(document.get("publicUrl"))
    if public_url:
        return [("publicUrl", public_url)]

    files = []
    for field in url_fields:
        value = _text(document.get(field))
        if value:
            files.append((field, value))
    return files


def _extract_pdf(file_path):
    reader = PdfReader(file_path)
    if reader.is_encrypted:
        reader.decrypt("")

    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text.strip())

    page_title = ""
    metadata = reader.metadata
    if metadata is not None and metadata.title:
        page_title = _text(metadata.title)

    return {
        "text": "\n\n".join(parts),
        "pages": len(reader.pages),
        "pageTitle": page_title,
    }


def _prepare_documents(resource_type, doc_ids, collection_name, collection_files):
    query = {"_id": {"$in": [ObjectId(doc_id) for doc_id in doc_ids]}}
    projection = {collection_files["title_field"]: 1}
    for field in collection_files["url_fields"]:
        projection[field] = 1

    found = {
        str(document["_id"]): document
        for document in database[collection_name].find(query, projection)
    }
    not_found = [doc_id for doc_id in doc_ids if doc_id not in found]
    prepared = []
    skipped = []

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    for doc_id in doc_ids:
        document = found.get(doc_id)
        if document is None:
            continue

        title = _document_title(document, collection_files["title_field"])
        files = _stored_files(document, collection_files["url_fields"])
        if not files:
            skipped.append({
                "documentId": doc_id,
                "error": "public url is missing",
            })
            continue

        for field, stored_value in files:
            source_title = title
            label = FILE_LABELS.get(field)
            if label and len(files) > 1:
                source_title = f"{title} - {label}" if title else label

            stored_path = None
            try:
                source_url, download_url = resolve_stored_file(stored_value)
                handle, stored_path = tempfile.mkstemp(
                    suffix=".pdf",
                    dir=UPLOAD_DIR,
                )
                os.close(handle)
                download_pdf(download_url, stored_path)
                extracted = _extract_pdf(stored_path)
            except Exception as exc:
                skipped.append({
                    "documentId": doc_id,
                    "field": field,
                    "error": _safe_error(exc),
                })
                continue
            finally:
                if stored_path and os.path.isfile(stored_path):
                    os.remove(stored_path)

            if not extracted["text"].strip():
                skipped.append({
                    "documentId": doc_id,
                    "field": field,
                    "error": "pdf has no extractable text",
                })
                continue

            prepared.append({
                "documentId": doc_id,
                "resourceType": resource_type,
                "sourceCollection": collection_name,
                "sourceUrl": source_url,
                "title": source_title,
                "slug": "",
                "pageTitle": extracted["pageTitle"],
                "pages": extracted["pages"],
                "text": extracted["text"],
            })

    return not_found, skipped, prepared


@public_pdf_bp.route("/pdf", methods=["POST"])
def upload_public_pdf_chunks():
    data = request_data()
    resource_type = _text(data.get("resourceType"))
    collection_name = _text(data.get("collection"))
    doc_ids = data.get("docIds") or []

    if not resource_type:
        return jsonify({"error": "resourceType is required"}), 400

    if not collection_name:
        return jsonify({"error": "collection is required"}), 400

    collection_files = COLLECTION_FILES.get(collection_name)
    if not collection_files:
        return jsonify({"error": "Unknown collection"}), 400

    if not isinstance(doc_ids, list) or not doc_ids:
        return jsonify({"error": "docIds must be a non-empty array"}), 400

    if not all(isinstance(doc_id, str) and doc_id.strip() for doc_id in doc_ids):
        return jsonify({"error": "docIds must contain valid strings"}), 400

    doc_ids = [doc_id.strip() for doc_id in doc_ids]
    invalid = [doc_id for doc_id in doc_ids if not ObjectId.is_valid(doc_id)]
    if invalid:
        return jsonify({"error": "Invalid docIds", "invalid": invalid}), 400

    try:
        not_found, skipped, prepared = _prepare_documents(
            resource_type,
            doc_ids,
            collection_name,
            collection_files,
        )
    except Exception:
        current_app.logger.exception("Public PDF lookup failed")
        return jsonify({"error": "Public PDF lookup failed"}), 500

    if not prepared:
        status = 404 if not skipped else 400
        message = "No documents found" if not skipped else "No PDFs could be read"
        return jsonify({
            "success": False,
            "resourceType": resource_type,
            "notFound": not_found,
            "skipped": skipped,
            "error": message,
        }), status

    try:
        summary = ingest_extracted_documents(
            prepared,
            database[MONGODB_DATA_CHUNKS_COLLECTION],
        )
    except Exception:
        current_app.logger.exception("Public PDF ingestion failed")
        return jsonify({"error": "Public PDF ingestion failed"}), 500

    stored = summary["inserted"]
    return jsonify({
        "success": stored > 0,
        "resourceType": resource_type,
        "requested": len(doc_ids),
        "storedChunks": stored,
        "notFound": not_found,
        "skipped": skipped,
        "results": summary["results"],
    }), 200
