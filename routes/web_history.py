from flask import Blueprint, jsonify, request

from rag_chain import get_web_history

# Create the blueprint with a name and URL prefix
web_history_bp = Blueprint("web-history" , __name__)

@web_history_bp.route("/web-history", methods=["GET"])
def get_web_history_route():
    data = request.args
    body = request.get_json(silent=True)
    if isinstance(body, dict):
        data = body

    user_id = data.get("userId") or ""
    print(user_id)

    if not isinstance(user_id, str) or not user_id.strip():
        return jsonify({"error": "userId is required"}), 400

    if len(user_id.strip()) > 200:
        return jsonify({"error": "userId is too long"}), 400

    try:
        return jsonify(get_web_history(user_id.strip()))
    except Exception:
        web_history_bp.logger.exception("Web history lookup failed")
        return jsonify({"error": "Unable to fetch history"}), 500