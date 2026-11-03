import ipaddress
import os
# import re
import socket
# import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from pymongo import MongoClient

from config import MONGODB_URI
from ingest import ingest_pdf
from rag_chain import (
    get_conversation,
    get_qa_chain,
    lecture_exists,
    save_conversation,
    save_question,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

UPLOAD_DIR = "uploads"
MAX_PDF_BYTES = 50 * 1024 * 1024

client = MongoClient(MONGODB_URI)

class PublicRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_public_http_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def validate_public_http_url(url):
    url = (url or "").strip()
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("url must be an http or https link")

    host = parsed.hostname.strip("[]").lower()
    if host == "localhost" or host.endswith(".local"):
        raise ValueError("url must be a public cloud link")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = socket.getaddrinfo(host, port)
    except socket.gaierror as exc:
        raise ValueError("url could not be resolved") from exc

    for info in addresses:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise ValueError("url must be a public cloud link")

    return url


def pdf_filename(url, headers):
    header_name = headers.get_filename() if headers is not None else None
    candidate = header_name or os.path.basename(unquote(urlparse(url).path))
    filename = secure_filename(candidate or "")

    if not filename.lower().endswith(".pdf"):
        return "document.pdf"

    return filename


def download_pdf(url, dest_path):
    url = validate_public_http_url(url)
    request = Request(url, headers={"User-Agent": "rag-pipeline"})
    opener = build_opener(PublicRedirectHandler)
    filename = "document.pdf"

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    try:
        with opener.open(request, timeout=60) as response:
            filename = pdf_filename(url, response.headers)
            with open(dest_path, "wb") as handle:
                total = 0
                checked_header = False

                while True:
                    block = response.read(64 * 1024)
                    if not block:
                        break

                    if not checked_header:
                        if b"%PDF" not in block[:1024]:
                            raise ValueError("url did not return a PDF")
                        checked_header = True

                    total += len(block)
                    if total > MAX_PDF_BYTES:
                        raise ValueError("pdf exceeds the 50 MB limit")

                    handle.write(block)

            if total == 0:
                raise ValueError("url returned an empty file")
    except ValueError:
        if os.path.isfile(dest_path):
            os.remove(dest_path)
        raise
    except (HTTPError, URLError, TimeoutError, socket.timeout) as exc:
        if os.path.isfile(dest_path):
            os.remove(dest_path)
        raise ValueError(f"could not download pdf: {exc}") from exc

    return filename


def request_data():
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return data

    return request.form


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "pdf exceeds the 50 MB limit"}), 413


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "RAG pipeline server is running",
        "documents": "POST /documents with JSON {\"url\": \"https://.../file.pdf\"}",
        "ask": "POST /ask with JSON {\"lectureId\": \"...\", \"userId\": \"...\", \"question\": \"...\"}",
        "history": "GET /history?lectureId=...&userId=...",
    })


@app.route("/documents", methods=["POST"])
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

    except ValueError as exc:

        if os.path.isfile(stored_path):
            os.remove(stored_path)

        return jsonify({"error": str(exc)}), 400

    except Exception as exc:

        if os.path.isfile(stored_path):
            os.remove(stored_path)

        app.logger.exception("PDF ingestion failed")

        return jsonify({"error": "PDF ingestion failed"}), 500

    summary["filename"] = filename
    summary["url"] = url
    summary["lectureId"] = lecture_id

    return jsonify(summary)



@app.route("/history", methods=["GET"])
def history():
    data = request.args
    body = request.get_json(silent=True)
    if isinstance(body, dict):
        data = body

    lecture_id = (
        data.get("lectureId")
        or data.get("vector_db_id")
        or ""
    ).strip()

    user_id = (data.get("userId") or "").strip()

    if not lecture_id:
        return jsonify({
            "error": "lectureId is required"
        }), 400

    if not user_id:
        return jsonify({
            "error": "user is required"
        }), 400

    try:
        if not lecture_exists(lecture_id):
            return jsonify({
                "error": "lecture not found"
            }), 404

        return jsonify(get_conversation(user_id, lecture_id))

    except Exception:
        app.logger.exception("History lookup failed")

        return jsonify({
            "error": "Unable to fetch history"
        }), 500


@app.route("/ask", methods=["POST"])
def ask():

    data = request_data()

    lecture_id = (
        data.get("lectureId")
        or data.get("vector_db_id")
        or ""
    ).strip()

    question = (data.get("question") or "").strip()

    user_id = (data.get("userId") or "").strip()

    if not lecture_id:
        return jsonify({
            "error": "lectureId is required"
        }), 400

    if not question:
        return jsonify({
            "error": "question is required"
        }), 400

    if not user_id:
        return jsonify({
            "error": "user is required"
        }), 400

    try:

        if not lecture_exists(lecture_id):
            return jsonify({
                "error": "lecture not found"
            }), 404

        chain, memory = get_qa_chain(user_id, lecture_id)

        save_question(user_id, lecture_id, question)

        result = chain.invoke({
            "question": question
        })

        save_conversation(user_id, lecture_id, memory)

        return jsonify({
            "lectureId": lecture_id,
            "question": question,
            "answer": result["answer"],
        })

    except Exception:

        app.logger.exception("RAG query failed")

        return jsonify({
            "error": "Unable to process question"
        }), 500



if __name__ == "__main__":
    try:
        client.admin.command("ping")
        print("MongoDB connected successfully")
    except Exception as e:
        raise RuntimeError(
            f"MongoDB connection failed: {e}"
        )

    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)
