from flask import Blueprint, jsonify

from server import request_data
from rag_chain import lecture_exists, save_question, get_qa_chain, save_conversation

# Create the blueprint with a name and URL prefix
lecture_ask_bp = Blueprint("ask-lecture" , __name__)

@lecture_ask_bp.route("/ask", methods=["POST"])
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

        lecture_ask_bp.logger.exception("RAG query failed")

        return jsonify({
            "error": "Unable to process question"
        }), 500
