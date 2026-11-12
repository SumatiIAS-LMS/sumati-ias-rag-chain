import hashlib
import hmac
import ipaddress
import os
import socket
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from pymongo import MongoClient

from config import (
    CF_ACCESS_KEY_ID,
    CF_S3_API_ENDPOINT,
    CF_S3_BUCKET_NAME,
    CF_SECRET_ACCESS_KEY,
    MONGODB_URI,
    MONGODB_WEB_CURRENT_AFFAIRS_COLLECTION,
    MONGODB_WEB_EDITORIAL_COLLECTION,
    MONGODB_WEB_HOT_TOPIC_COLLECTION,
    PUBLIC_ASSET_BASE_URL,
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
    from routes.upload_public_pdf_chunks import public_pdf_bp
    from routes.lecture_ask import lecture_ask_bp
    from routes.lecture_chat_history import lecture_chat_history_bp
    from routes.web_ask import web_ask_bp

    app.register_blueprint(web_history_bp)
    app.register_blueprint(home_bp)
    app.register_blueprint(web_page_bp)
    app.register_blueprint(public_pdf_bp)
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


def _sign(key, message):
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def _r2_settings():
    access_key = (CF_ACCESS_KEY_ID or "").strip()
    secret_key = (CF_SECRET_ACCESS_KEY or "").strip()
    endpoint = (CF_S3_API_ENDPOINT or "").strip().rstrip("/")
    bucket = (CF_S3_BUCKET_NAME or "").strip()
    parsed = urlparse(endpoint)

    if (
        not access_key
        or not secret_key
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not bucket
    ):
        return None

    return access_key, secret_key, endpoint, bucket, parsed.netloc


def _object_uri(bucket, key):
    return "/" + "/".join(
        quote(part, safe="")
        for part in [bucket, *key.strip("/").split("/")]
    )


def _signing_key(secret_key, datestamp, region, service):
    signing_key = _sign(("AWS4" + secret_key).encode("utf-8"), datestamp)
    signing_key = _sign(signing_key, region)
    signing_key = _sign(signing_key, service)
    return _sign(signing_key, "aws4_request")


def presign_r2_get(key, expires=3600):
    """Build a temporary download URL for an object key in the asset bucket."""
    settings = _r2_settings()
    if settings is None:
        raise ValueError("stored file has no public url")

    access_key, secret_key, endpoint, bucket, host = settings

    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = now.strftime("%Y%m%d")
    region = "auto"
    service = "s3"
    credential_scope = f"{datestamp}/{region}/{service}/aws4_request"
    canonical_uri = _object_uri(bucket, key)
    params = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{access_key}/{credential_scope}",
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": str(expires),
        "X-Amz-SignedHeaders": "host",
    }
    canonical_query = "&".join(
        f"{quote(name, safe='')}={quote(params[name], safe='-')}"
        for name in sorted(params)
    )
    canonical_headers = f"host:{host}\n"
    canonical_request = "\n".join([
        "GET",
        canonical_uri,
        canonical_query,
        canonical_headers,
        "host",
        "UNSIGNED-PAYLOAD",
    ])
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])
    signing_key = _signing_key(secret_key, datestamp, region, service)
    signature = hmac.new(
        signing_key,
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return f"{endpoint}{canonical_uri}?{canonical_query}&X-Amz-Signature={signature}"


def r2_key_exists(key):
    """Return whether this object key is present in the asset bucket."""
    key = (key or "").strip()
    settings = _r2_settings()
    if not key or settings is None:
        return False

    access_key, secret_key, endpoint, bucket, host = settings
    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = now.strftime("%Y%m%d")
    region = "auto"
    service = "s3"
    payload_hash = "UNSIGNED-PAYLOAD"
    credential_scope = f"{datestamp}/{region}/{service}/aws4_request"
    canonical_uri = _object_uri(bucket, key)
    canonical_headers = (
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
    )
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_request = "\n".join([
        "HEAD",
        canonical_uri,
        "",
        canonical_headers,
        signed_headers,
        payload_hash,
    ])
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])
    signature = hmac.new(
        _signing_key(secret_key, datestamp, region, service),
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    authorization = (
        "AWS4-HMAC-SHA256 "
        f"Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )
    request = Request(
        f"{endpoint}{canonical_uri}",
        method="HEAD",
        headers={
            "Host": host,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
            "Authorization": authorization,
        },
    )

    try:
        with build_opener().open(request, timeout=15) as response:
            return 200 <= response.status < 300
    except (HTTPError, URLError, TimeoutError, socket.timeout):
        return False


def signed_source_url(stored_value, cache=None):
    """Sign an object key after confirming it exists. Leave web links unchanged."""
    stored_value = (stored_value or "").strip()
    if not stored_value or stored_value.lower().startswith(("http://", "https://")):
        return stored_value

    if cache is not None and stored_value in cache:
        return cache[stored_value]

    signed = stored_value
    if r2_key_exists(stored_value):
        try:
            signed = presign_r2_get(stored_value)
        except ValueError:
            signed = stored_value

    if cache is not None:
        cache[stored_value] = signed
    return signed


def resolve_stored_file(stored_value):
    """Return the stable source url and the url that can be downloaded."""
    stored_value = (stored_value or "").strip()
    if not stored_value:
        raise ValueError("public url is missing")

    if stored_value.lower().startswith(("http://", "https://")):
        return stored_value, stored_value

    if PUBLIC_ASSET_BASE_URL:
        public_url = PUBLIC_ASSET_BASE_URL.rstrip("/") + "/" + quote(
            stored_value.strip("/"),
            safe="/",
        )
        return public_url, public_url

    return stored_value, presign_r2_get(stored_value)


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
