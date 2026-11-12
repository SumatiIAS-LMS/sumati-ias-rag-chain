import ipaddress
import os
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from pymongo import MongoClient

from config import (
    MONGODB_URI,
    MONGODB_WEB_CURRENT_AFFAIRS_COLLECTION,
    MONGODB_WEB_EDITORIAL_COLLECTION,
    MONGODB_WEB_HOT_TOPIC_COLLECTION
)

from waitress import serve

def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

    @app.errorhandler(413)
    def file_too_large():
        return jsonify({"error": "pdf exceeds the 50 MB limit"}), 413

    @app.errorhandler(404)
    def not_found():
        return jsonify({"error": "resource not found"}), 404

    from routes.web_history import web_history_bp
    from routes.home import home_bp
    from routes.upload_documents import upload_documents_bp
    from routes.upload_webpage_chunks import web_page_bp
    from routes.lecture_ask import lecture_ask_bp
    from routes.lecture_chat_history import lecture_chat_history_bp
    from routes.web_ask import web_ask_bp

    app.register_blueprint(web_history_bp)
    app.register_blueprint(home_bp)
    app.register_blueprint(web_page_bp)
    app.register_blueprint(upload_documents_bp)
    app.register_blueprint(lecture_ask_bp)
    app.register_blueprint(lecture_chat_history_bp)
    app.register_blueprint(web_ask_bp)

    return app


UPLOAD_DIR = "uploads"
MAX_PDF_BYTES = 50 * 1024 * 1024

RESOURCE_CONFIG = {
    "current-affair": (MONGODB_WEB_CURRENT_AFFAIRS_COLLECTION, "title"),
    "web-hot-topic": (MONGODB_WEB_HOT_TOPIC_COLLECTION, "title"),        # change if it has its own collection
    "web-editorial": (MONGODB_WEB_EDITORIAL_COLLECTION, "editorialTitle"),  # change if it has its own collection
}

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


if __name__ == "__main__":
    try:
        client.admin.command("ping")
        print("MongoDB connected successfully")
    except Exception as e:
        raise RuntimeError(
            f"MongoDB connection failed: {e}"
        )

    serve(
        create_app(),
        host="127.0.0.1",
        port=8000,
        threads=8
    )
