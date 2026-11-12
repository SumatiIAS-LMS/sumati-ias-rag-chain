from flask import Blueprint, jsonify

from server import request_data, UPLOAD_DIR, download_pdf
from ingest import ingest_pdf
from werkzeug.utils import secure_filename

import os

# Create the blueprint with a name and URL prefix
upload_documents_bp = Blueprint("upload-documents" , __name__)


@upload_documents_bp.route("/documents", methods=["POST"])
def upload_document():

    data = request_data()

    url = (data.get("url") or "").strip()
    lecture_id = (data.get("lectureId") or "").strip()

    if not url:
        return jsonify({"error": "url is required"}), 400

    if not lecture_id:
        return jsonify({"error": "lectureId is required"}), 400

    stored_path = os.path.join(
        UPLOAD_DIR,
        f"{secure_filename(lecture_id)}.pdf"
    )

    try:

        filename = download_pdf(url, stored_path)

        summary = ingest_pdf(
            stored_path,
            lecture_id
        )

        summary["filename"] = filename
        summary["url"] = url
        summary["lectureId"] = lecture_id
    
        return jsonify(summary)

    except ValueError as exc:

        if os.path.isfile(stored_path):
            os.remove(stored_path)

        return jsonify({"error": str(exc)}), 400

    except Exception as exc:

        if os.path.isfile(stored_path):
            os.remove(stored_path)

        upload_documents_bp.logger.exception("PDF ingestion failed")

        return jsonify({"error": "PDF ingestion failed"}), 500

    finally:
        if os.path.isfile(stored_path):
            os.remove(stored_path)
            upload_documents_bp.logger.info(
                "Temporary PDF deleted: %s",
                stored_path
            )
