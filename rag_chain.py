from langchain_classic.chains import ConversationalRetrievalChain, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from pymongo import ASCENDING, MongoClient, ReturnDocument
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime, timezone, timedelta
from langchain_core.runnables import RunnableLambda

from bson import ObjectId

from config import (
    LLM_MODEL,
    EMBEDDING_MODEL_NAME,
    MONGODB_URI,
    MONGODB_DATABASE,
    MONGODB_COLLECTION,
    MONGODB_DATA_CHUNKS_COLLECTION,
    VECTOR_INDEX_NAME,
    SUMATI_IAS_DATA_CHUNKS_VECTOR_INDEX_NAME,
    CONTEXT_EXPIREY,
    WEB_CONTEXT_EXPIRY,
    WEB_DAILY_MESSAGE_LIMIT,
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

web_chat_collection = database["web_chat_history"]

web_chat_collection.create_index("userId", unique=True)

# Replace an older TTL index so the 24-hour expiry can be created
_web_expiry_index = web_chat_collection.index_information().get("expiresAt_1")
if _web_expiry_index and _web_expiry_index.get("expireAfterSeconds") != WEB_CONTEXT_EXPIRY:
    web_chat_collection.drop_index("expiresAt_1")

# Automatically delete expired browser conversations after 24 hours
web_chat_collection.create_index(
    [("expiresAt", ASCENDING)],
    expireAfterSeconds=WEB_CONTEXT_EXPIRY
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


def get_public_data_vector_store():
    """Vector store for scraped web chunks used by the chatbot."""
    return MongoDBAtlasVectorSearch(
        collection=database[MONGODB_DATA_CHUNKS_COLLECTION],
        embedding=get_embeddings(),
        index_name=SUMATI_IAS_DATA_CHUNKS_VECTOR_INDEX_NAME,
        text_key="text",
        embedding_key="embedding",
        relevance_score_fn="cosine",
    )



_WEB_QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an AI assistant for Sumati IAS.

Your primary responsibility is to answer students' questions using the
resources provided by Sumati IAS. However, you may use external sources
when the required information is not available in those resources.

Follow this priority strictly:

1. FIRST PRIORITY - PROVIDED RESOURCES:
   - Carefully examine the provided resources and retrieved context.
   - Identify information directly relevant to the student's question.
   - Give preference to information from these resources.
   - Do not ignore relevant information just because external knowledge
     provides a different or more detailed explanation.

2. SECOND PRIORITY - CONVERSATION HISTORY:
   - Examine previous messages to understand the current question,
     references, and follow-up queries.
   - Use earlier messages wherever they provide relevant information.
   - Do not ask the student to repeat information already available
     in the conversation.

3. THIRD PRIORITY - EXTERNAL INFORMATION:
   - Only when the provided resources, retrieved context, and previous
     messages do not contain sufficient information to answer the
     question, use relevant information from outside the provided
     resources.
   - External information may be used to supplement incomplete answers
     or answer questions that are not covered in the provided resources.
   - Prefer reliable, authoritative, and up-to-date sources.
   - For UPSC-related topics, prioritize official government sources,
     government reports, constitutional provisions, legislation,
     parliamentary publications, and reputed institutional sources.
   - Never assume that information is available in external sources
     unless it has actually been retrieved or supplied to you.

4. ANSWERING RULES:
   - Never invent facts, statistics, dates, quotations, or references.
   - If information from the provided resources is sufficient, answer
     using those resources without unnecessarily introducing external
     information.
   - If external information is required, use it only for the missing
     portions of the answer.
   - If the available resources and external information still do not
     provide a reliable answer, clearly state that the information
     could not be verified.
   - If provided resources and reliable external information conflict,
     prioritize the provided resources when explaining their content,
     but mention the discrepancy when it is important.
   - Do not mention internal retrieval processes, vector databases,
     embeddings, or retrieved context.
   - Do not claim that an external search was performed unless
     external information was actually provided to you.

5. RESPONSE STYLE:
   - Keep responses accurate, concise, clear, and student-friendly.
   - Explain concepts in a manner suitable for UPSC aspirants.
   - Use structured headings, bullet points, and examples whenever
     they improve understanding.
   - Maintain continuity with the previous conversation.

Provided Context:
{context}
        """,
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


_STANDALONE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the latest user question as one standalone question using the chat history. Return only the rewritten question.",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


class WebMessageLimitError(Exception):
    def __init__(self, expires_at):
        self.expires_at = expires_at
        super().__init__("daily message limit reached")


def _utc(value):
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _history_messages(messages):
    converted = []
    for message in messages:
        content = message.get("content") or ""
        if message.get("role") == "human":
            converted.append(HumanMessage(content=content))
        elif message.get("role") == "ai":
            converted.append(AIMessage(content=content))
    return converted


def _standalone_question(question, history):
    if not history:
        return question

    rewritten = (
        _STANDALONE_QUESTION_PROMPT
        | ChatOpenAI(model=LLM_MODEL, temperature=0)
        | StrOutputParser()
    ).invoke({
        "chat_history": history,
        "input": question,
    })
    return rewritten.strip() or question


def begin_web_chat(user_id):
    """Reserve one user message in this browser's one-day conversation window."""
    now = datetime.now(timezone.utc)
    chat = web_chat_collection.find_one({"userId": user_id})
    active = chat is not None and _utc(chat.get("expiresAt")) > now

    if active and chat.get("userMessageCount", 0) >= WEB_DAILY_MESSAGE_LIMIT:
        raise WebMessageLimitError(_utc(chat.get("expiresAt")))

    if active:
        updated = web_chat_collection.find_one_and_update(
            {
                "userId": user_id,
                "expiresAt": chat["expiresAt"],
                "userMessageCount": {"$lt": WEB_DAILY_MESSAGE_LIMIT},
            },
            {
                "$inc": {"userMessageCount": 1},
                "$set": {"updatedAt": now},
            },
            return_document=ReturnDocument.AFTER,
        )
        if updated is None:
            raise WebMessageLimitError(_utc(chat.get("expiresAt")))
        return {
            "messages": chat.get("messages") or [],
            "expiresAt": updated["expiresAt"],
            "userMessageCount": updated["userMessageCount"],
        }

    expires_at = now + timedelta(seconds=WEB_CONTEXT_EXPIRY)
    updated = web_chat_collection.find_one_and_update(
        {"userId": user_id},
        {
            "$set": {
                "messages": [],
                "userMessageCount": 1,
                "createdAt": now,
                "updatedAt": now,
                "expiresAt": expires_at,
            }
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return {
        "messages": [],
        "expiresAt": updated["expiresAt"],
        "userMessageCount": updated["userMessageCount"],
    }


def save_web_chat_turn(user_id, expires_at, question, answer, sources):
    now = datetime.now(timezone.utc)
    web_chat_collection.update_one(
        {"userId": user_id, "expiresAt": expires_at},
        {
            "$push": {
                "messages": {
                    "$each": [
                        {"role": "human", "content": question, "createdAt": now},
                        {"role": "ai", "content": answer, "sources": sources, "createdAt": now},
                    ]
                }
            },
            "$set": {"updatedAt": now},
        },
    )


def release_web_chat_slot(user_id, expires_at):
    web_chat_collection.update_one(
        {
            "userId": user_id,
            "expiresAt": expires_at,
            "userMessageCount": {"$gt": 0},
        },
        {"$inc": {"userMessageCount": -1}},
    )


def get_web_history(user_id):
    """Return the active one-day conversation for a browser, if it has not expired."""
    now = datetime.now(timezone.utc)
    chat = web_chat_collection.find_one({"userId": user_id})
    if chat is None or _utc(chat.get("expiresAt")) <= now:
        return {
            "userId": user_id,
            "messages": [],
            "messagesUsed": 0,
            "messagesRemaining": WEB_DAILY_MESSAGE_LIMIT,
            "expiresAt": None,
            "updatedAt": None,
        }

    messages = []
    for message in chat.get("messages") or []:
        created_at = message.get("createdAt")
        messages.append({
            "role": message.get("role") or "",
            "content": message.get("content") or "",
            "sources": message.get("sources") or [],
            "createdAt": _utc(created_at).isoformat() if created_at else None,
        })

    used = chat.get("userMessageCount") or 0
    updated_at = chat.get("updatedAt")
    return {
        "userId": user_id,
        "messages": messages,
        "messagesUsed": used,
        "messagesRemaining": max(WEB_DAILY_MESSAGE_LIMIT - used, 0),
        "expiresAt": _utc(chat.get("expiresAt")).isoformat(),
        "updatedAt": _utc(updated_at).isoformat() if updated_at else None,
    }


def web_chunks_exist():
    return database[MONGODB_DATA_CHUNKS_COLLECTION].count_documents({}, limit=1) > 0


def _web_source(document):
    metadata = document.metadata or {}
    document_id = metadata.get("documentId")
    return {
        "documentId": "" if document_id is None else str(document_id),
        "resourceType": metadata.get("resourceType") or "",
        "sourceUrl": metadata.get("sourceUrl") or "",
        "title": metadata.get("title") or metadata.get("pageTitle") or "",
        "chunkIndex": metadata.get("chunkIndex"),
    }



def answer_web_question(question, chat_history=None):
    """Answer from all scraped web chunks, using this browser's saved chat only."""

    history = _history_messages(chat_history or [])
    search_question = _standalone_question(question, history)

    retriever = get_public_data_vector_store().as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    def retrieve(inputs):
        return retriever.invoke(
            inputs.get("search_input") or inputs["input"]
        )

    # Convert normal Python function into a LangChain Runnable
    retrieve_runnable = RunnableLambda(retrieve)

    combine_docs_chain = create_stuff_documents_chain(
        ChatOpenAI(model=LLM_MODEL, temperature=0),
        _WEB_QA_PROMPT,
    )

    result = create_retrieval_chain(
        retrieve_runnable,
        combine_docs_chain
    ).invoke({
        "input": question,
        "search_input": search_question,
        "chat_history": history,
    })

    answer = result.get("answer") or ""

    if not isinstance(answer, str):
        answer = getattr(answer, "content", str(answer))

    sources = []

    for document in result.get("context") or []:
        source = _web_source(document)
        if not any(existing["documentId"] == source["documentId"] for existing in sources):
            sources.append(source)

    return {
        "answer": answer,
        "sources": sources,
    }


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