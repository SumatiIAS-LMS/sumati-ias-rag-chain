from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community import FastEmbedEmbeddings
from pymongo import MongoClient, ASCENDING
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime, timezone, timedelta

from bson import ObjectId

from config import (
    LLM_MODEL,
    EMBEDDING_MODEL_NAME,
    MONGODB_URI,
    MONGODB_DATABASE,
    MONGODB_COLLECTION,
    VECTOR_INDEX_NAME,
    CONTEXT_EXPIREY
)

mongo_client = MongoClient(MONGODB_URI)

database = mongo_client[MONGODB_DATABASE]

collection = database[MONGODB_COLLECTION]

chat_collection = database["rag_chat_history"]

# One conversation per user per lecture
chat_collection.create_index(
    [("userId", ASCENDING), ("lectureId", ASCENDING)],
    unique=True
)

# Automatically delete expired conversations
chat_collection.create_index(
    [("expiresAt", ASCENDING)],
    expireAfterSeconds=CONTEXT_EXPIREY
)

_embeddings = None

def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            max_length=512,
        )

    return _embeddings

def get_vector_store():

    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=get_embeddings(),
        index_name=VECTOR_INDEX_NAME,
        text_key="text",
        embedding_key="embedding",
        relevance_score_fn="cosine",
    )


def build_rag_chain(lecture_id, memory):

    vectorstore = get_vector_store()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4,
            "pre_filter": {
                "lectureId": {
                    "$eq": ObjectId(lecture_id)
                }
            },
        },
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
    )

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
            You are an AI Lecture Assistant. Answer using the provided context as your primary reference. Do not mention the context, retrieved documents, sources, or where the information came from.

            If the required information is missing or partially covered, supplement it using reliable online sources without mentioning the retrieval process. Never invent facts or present uncertain information as factual.

            Keep responses accurate, concise, clear, and student-friendly.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": qa_prompt
        }
    )


def _conversation_filter(user_id, lecture_id):
    return {
        "userId": ObjectId(user_id),
        "lectureId": ObjectId(lecture_id),
    }


def _conversation_timestamps(now):
    return {
        "updatedAt": now,
        "expiresAt": now + timedelta(seconds=CONTEXT_EXPIREY),
    }


def save_question(user_id, lecture_id, question):
    now = datetime.now(timezone.utc)

    chat_collection.update_one(
        _conversation_filter(user_id, lecture_id),
        {
            "$push": {
                "messages": {
                    "role": "human",
                    "content": question,
                }
            },
            "$set": _conversation_timestamps(now),
        },
        upsert=True,
    )


def save_conversation(user_id, lecture_id, memory):

    now = datetime.now(timezone.utc)

    messages = []

    for msg in memory.chat_memory.messages:
        if isinstance(msg, HumanMessage):
            messages.append({
                "role": "human",
                "content": msg.content
            })

        elif isinstance(msg, AIMessage):
            messages.append({
                "role": "ai",
                "content": msg.content
            })

    chat_collection.update_one(
        _conversation_filter(user_id, lecture_id),
        {
            "$set": {
                "messages": messages,
                **_conversation_timestamps(now),
            }
        },
        upsert=True
    )


def get_conversation(user_id, lecture_id):
    conversation = chat_collection.find_one(
        _conversation_filter(user_id, lecture_id),
        {"_id": 0, "messages": 1, "updatedAt": 1},
    )

    if not conversation:
        return {
            "userId": str(user_id),
            "lectureId": str(lecture_id),
            "messages": [],
        }

    updated_at = conversation.get("updatedAt")

    return {
        "userId": str(user_id),
        "lectureId": str(lecture_id),
        "messages": conversation.get("messages", []),
        "updatedAt": updated_at.isoformat() if updated_at else None,
    }


def get_qa_chain(user_id, lecture_id):
    conversation = chat_collection.find_one({
        "userId": ObjectId(user_id),
        "lectureId": ObjectId(lecture_id)
    })

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
    )

    # Restore previous conversation
    if conversation:
        print("conversation ", conversation)
        for msg in conversation.get("messages", []):
            if msg["role"] == "human":
                memory.chat_memory.add_message(
                    HumanMessage(content=msg["content"])
                )
            elif msg["role"] == "ai":
                memory.chat_memory.add_message(
                    AIMessage(content=msg["content"])
                )

    return build_rag_chain(lecture_id, memory), memory

def lecture_exists(lecture_id):

    return collection.count_documents({
        "lectureId": ObjectId(lecture_id)
    }) > 0