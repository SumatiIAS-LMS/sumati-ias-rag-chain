from flask import Blueprint, jsonify, request

from rag_chain import lecture_exists, get_conversation

# Create the blueprint with a name and URL prefix
lecture_chat_history_bp = Blueprint("lecture-history" , __name__)

@lecture_chat_history_bp.route("/history", methods=["GET"])
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
        lecture_chat_history_bp.logger.exception("History lookup failed")

        return jsonify({
            "error": "Unable to fetch history"
        }), 500
