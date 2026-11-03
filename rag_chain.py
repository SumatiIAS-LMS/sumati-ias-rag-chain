from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
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
    VECTOR_INDEX_NAME
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
    "expiresAt",
    expireAfterSeconds=10
)

_embeddings = None
_chains = {}


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
                    "$eq": lecture_id
                }
            },
        },
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
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
        {
            "userId": str(user_id),
            "lectureId": str(lecture_id)
        },
        {
            "$set": {
                "messages": messages,
                "updatedAt": now,
                "expiresAt": now + timedelta(days=2)
            }
        },
        upsert=True
    )


def get_qa_chain(user_id, lecture_id):
    conversation = chat_collection.find_one({
        "userId": str(user_id),
        "lectureId": str(lecture_id)
    })

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
    )

    # Restore previous conversation
    if conversation:
        for msg in conversation.get("messages", []):
            if msg["role"] == "human":
                memory.chat_memory.add_message(
                    HumanMessage(content=msg["content"])
                )
            elif msg["role"] == "ai":
                memory.chat_memory.add_message(
                    AIMessage(content=msg["content"])
                )
        return build_rag_chain(lecture_id, memory)

def lecture_exists(lecture_id):

    return collection.count_documents({
        "lectureId": ObjectId(lecture_id)
    }) > 0