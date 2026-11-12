from flask import Blueprint, jsonify

# Create the blueprint with a name and URL prefix
home_bp = Blueprint("home_bp" , __name__)

@home_bp.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "RAG pipeline server is running",
        "documents": "POST /documents with JSON {\"url\": \"https://.../file.pdf\"}",
        "webPage": "POST /web-page with JSON {\"resourceType\": \"...\", \"baseUrl\": \"https://.../\", \"docIds\": [\"...\"]}",
        "pdf": "POST /pdf with JSON {\"resourceType\": \"PRELIMS_STRIKER\", \"collection\": \"strikers\", \"docIds\": [\"...\"]}",
        "ask": "POST /ask with JSON {\"lectureId\": \"...\", \"userId\": \"...\", \"question\": \"...\"}",
        "askWeb": "POST /ask-web with JSON {\"userId\": \"...\", \"question\": \"...\"}",
        "webHistory": "GET /web-history?userId=...",
        "history": "GET /history?lectureId=...&userId=...",
    })