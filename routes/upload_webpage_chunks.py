from flask import Blueprint, jsonify
from web_scrapper import ingest_web_pages

from server import request_data, RESOURCE_CONFIG
from bson import ObjectId
from rag_chain import database
from config import MONGODB_DATA_CHUNKS_COLLECTION

# Create the blueprint with a name and URL prefix
web_page_bp = Blueprint("web-page" , __name__)


@web_page_bp.route("/web-page", methods=["POST"])
def upload_webpage_chunks():
    data = request_data()
    
    resourceType = (data.get("resourceType") or "").strip()
    docIds = data.get("docIds") or []
    baseUrl = (data.get("baseUrl") or "").strip()

    if not resourceType:
        return jsonify({"error": "resourceType is required"}), 400

    if not baseUrl:
        return jsonify({"error": "baseUrl is required"}), 400

    if not isinstance(docIds, list) or not docIds:
        return jsonify({"error": "docIds must be a non-empty array"}), 400

    if not all(isinstance(d, str) and d.strip() for d in docIds):
        return jsonify({"error": "docIds must contain valid strings"}), 400

    docIds = [d.strip() for d in docIds]

    invalid = [d for d in docIds if not ObjectId.is_valid(d)]
    if invalid:
        return jsonify({"error": "Invalid docIds", "invalid": invalid}), 400

    config = RESOURCE_CONFIG.get(resourceType)
    if not config:
        return jsonify({"error": "Invalid resourceType"}), 400

    collection_name, title_field = config
    collection = database[collection_name]

    cursor = collection.find(
        {"_id": {"$in": [ObjectId(d) for d in docIds]}},
        {title_field: 1, "slug": 1},
    )
    found = {str(doc["_id"]): doc for doc in cursor}
    not_found = [doc_id for doc_id in docIds if doc_id not in found]

    pages = []
    skipped = []

    for doc_id in docIds:
        if doc_id not in found:
            continue

        doc = found[doc_id]
        slug = (doc.get("slug") or "").strip()
        title = doc.get(title_field) or ""
        if not isinstance(title, str):
            title = str(title)

        if not slug:
            skipped.append({
                "documentId": doc_id,
                "error": "slug is missing",
            })
            continue

        try:
            source_url = baseUrl + "/" + slug
        except ValueError as exc:
            skipped.append({
                "documentId": doc_id,
                "slug": slug,
                "error": str(exc),
            })
            continue

        pages.append({
            "url": source_url,
            "documentId": doc_id,
            "resourceType": resourceType,
            "sourceCollection": collection_name,
            "title": title,
            "slug": slug,
        })

    if not pages:
        status = 404 if not skipped else 400
        message = "No documents found" if not skipped else "No pages could be scraped"
        return jsonify({
            "success": False,
            "resourceType": resourceType,
            "notFound": not_found,
            "skipped": skipped,
            "error": message,
        }), status

    try:
        summary = ingest_web_pages(
            pages,
            database[MONGODB_DATA_CHUNKS_COLLECTION],
        )
    except Exception:
        upload_webpage_chunks.logger.exception("Web page ingestion failed")
        return jsonify({"error": "Web page ingestion failed"}), 500

    stored = summary["inserted"]
    return jsonify({
        "success": stored > 0,
        "resourceType": resourceType,
        "requested": len(docIds),
        "storedChunks": stored,
        "notFound": not_found,
        "skipped": skipped,
        "results": summary["results"],
    }), 200