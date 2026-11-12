from flask import Blueprint, jsonify
from server import request_data
from rag_chain import (
    WebMessageLimitError,
    _utc,
    answer_web_question,
    begin_web_chat,
    release_web_chat_slot,
    save_web_chat_turn,
    web_chunks_exist,
)

from config import WEB_DAILY_MESSAGE_LIMIT

# Create the blueprint with a name and URL prefix
web_ask_bp = Blueprint("ask-web" , __name__)

@web_ask_bp.route("/ask-web", methods=["POST"])
def ask_web():
    data = request_data()

    question = data.get("question") or ""
    user_id = data.get("userId") or ""

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "question is required"}), 400

    if not isinstance(user_id, str) or not user_id.strip():
        return jsonify({"error": "userId is required"}), 400

    if len(user_id.strip()) > 200:
        return jsonify({"error": "userId is too long"}), 400

    question = question.strip()
    user_id = user_id.strip()

    try:
        if not web_chunks_exist():
            return jsonify({"error": "no matching content found"}), 404

        slot = begin_web_chat(user_id)
        try:
            result = answer_web_question(
                question,
                chat_history=slot["messages"],
            )
            save_web_chat_turn(
                user_id,
                slot["expiresAt"],
                question,
                result["answer"],
                result["sources"]
            )
        except Exception:
            release_web_chat_slot(user_id, slot["expiresAt"])
            raise

        used = slot["userMessageCount"]
        return jsonify({
            "question": question,
            "userId": user_id,
            "answer": result["answer"],
            "sources": result["sources"],
            "messagesUsed": used,
            "messagesRemaining": WEB_DAILY_MESSAGE_LIMIT - used,
            "expiresAt": _utc(slot["expiresAt"]).isoformat(),
        })

    except WebMessageLimitError as exc:
        return jsonify({
            "error": "daily message limit reached",
            "limit": WEB_DAILY_MESSAGE_LIMIT,
            "userId": user_id,
            "expiresAt": exc.expires_at.isoformat() if exc.expires_at else None,
        }), 429

    except Exception:
        web_ask_bp.logger.exception("Web RAG query failed")
        return jsonify({"error": "Unable to process question"}), 500