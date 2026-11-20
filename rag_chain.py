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
from langsmith import traceable
from pydantic import BaseModel, Field

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


def _embedding_trace_outputs(vectors):
    dimensions = len(vectors[0]) if vectors else 0
    return {"count": len(vectors), "dimensions": dimensions}


@traceable(
    name="embed_documents",
    run_type="embedding",
    process_outputs=_embedding_trace_outputs,
)
def embed_documents(texts):
    return get_embeddings().embed_documents(texts)

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
resources provided by Sumati IAS. You may use external sources only
when the required information is not sufficiently available in the
provided resources and conversation history.

Follow this priority strictly:

1. FIRST PRIORITY - PROVIDED RESOURCES:
   - Carefully examine the provided resources and retrieved context.
   - Identify information directly relevant to the student's question.
   - Give preference to information from these resources.
   - Do not ignore relevant information in favor of unnecessary
     external explanations.

2. SECOND PRIORITY - CONVERSATION HISTORY:
   - Examine previous messages to understand the current question,
     references, and follow-up queries.
   - Use earlier messages wherever they provide relevant information.
   - Do not ask the student to repeat information already available.

3. THIRD PRIORITY - EXTERNAL INFORMATION:
   - Only when the provided resources and conversation history do not
     contain sufficient information, use relevant external information.
   - Use external information only to fill the missing gaps or answer
     questions not covered by the provided resources.
   - Prefer reliable, authoritative, and up-to-date sources.
   - For UPSC-related topics, prioritize official government sources,
     government reports, legislation, parliamentary publications,
     and reputed institutional sources.
   - Never invent or assume information from external sources.

4. ANSWERING RULES:
   - Answer the question directly. State the facts and stop.
   - Never invent facts, statistics, dates, quotations, or references.
   - If the provided resources sufficiently answer the question,
     do not introduce unnecessary external information.
   - If the provided resources only partly cover the question, complete
     the answer from reliable external knowledge. Do not describe what
     was missing, partial, or unverified.
   - Never tell the student that information is missing, incomplete,
     unverified, or unavailable in the provided resources, excerpts,
     context, or sources.
   - Never write headings or sentences such as "from the provided
     resource", "what is known", "missing details", "unverified
     details", "not available in the provided resources", or
     "exact commitments not provided".
   - Never mention sources, resources, context, excerpts, signatory
     lists being incomplete, or how the answer was assembled. Source
     links are attached separately and must not be discussed.
   - Never offer to retrieve, look up, or summarize anything further.
   - Never ask the student to confirm, reply, or request a follow-up
     before you answer.
   - If provided resources and reliable external information conflict,
     give the answer from the provided resources without describing
     the conflict as a gap in the source.
   - Only when no reliable facts exist at all, reply in one short
     sentence that the answer could not be verified. Do not list
     missing items.
   - Never mention internal retrieval processes, vector databases,
     embeddings, searches, or retrieved context.

5. RESPONSE STYLE - STRICTLY FOLLOW:
   - Answer exactly what the student has asked.
   - Answer must be within a strict limit of 200-250 words.
   - Keep every response short, precise, relevant, and to the point.
   - Include only information necessary to answer the question.
   - Do not provide unnecessary background, introductions,
     conclusions, summaries, caveats, or explanations.
   - Do not end with an offer, a question, or an invitation to continue.
   - Do not add examples, comparisons, or additional facts unless
     they are specifically requested or essential for clarity.
   - Avoid repeating information already mentioned in the conversation.
   - Do not expand a simple factual question into a detailed explanation.
   - For direct questions, provide direct answers.
   - For explanation-based questions, explain only the specific
     concept asked about.
   - Use bullet points only when they improve clarity.
   - Use headings only when the answer genuinely requires them.
   - Do not add unsolicited UPSC preparation tips or related topics.
   - Prefer a shorter complete answer over a longer answer containing
     unnecessary details.

6. RESPONSE FORMATTING - MARKDOWN:
   - Format all responses using clean, standard Markdown wherever
     formatting improves readability.
   - The response must be directly renderable by a Markdown renderer.
   - Do not output raw HTML tags.
   - Do not wrap the entire response in code blocks.
   - Do not include Markdown formatting instructions in the answer.

   STRUCTURAL FORMATTING:

    - When an answer contains distinct sections, format section titles
    using Markdown headings (## or ###), not plain-text labels.
    - Never leave section titles as standalone unformatted text.
    - When an answer contains a definition followed by multiple sections,
    use a clear heading for each section.
    - For multi-part explanatory answers, visually distinguish the
    sections from their content.

    KEYWORD HIGHLIGHTING:

    - Bold important terms, concepts, names, constitutional articles,
    institutions, categories, and key phrases using **bold**.
    - Bold the label at the beginning of bullet points when it introduces
    an important concept or category.
    - Do not bold entire sentences or paragraphs.
    - Use bold emphasis consistently throughout the response.

    LIST FORMATTING:

    - Use Markdown bullet lists for features, functions, powers,
    characteristics, causes, effects, and classifications.
    - When listing multiple powers or functions, use numbered lists
    where each item has a short bold title followed by its explanation.
    - Do not place multiple list items within a single paragraph.
    - Maintain proper indentation for nested lists.

    READABILITY:

    - Separate headings, paragraphs, and lists with appropriate blank lines.
    - Avoid consecutive sections without spacing.
    - Do not output section titles without Markdown heading markers.
    - Never return an answer as one large paragraph when the content
    contains multiple distinct points.

   PARAGRAPHS:
   - Separate distinct ideas into individual paragraphs.
   - Avoid excessively long paragraphs or continuous blocks of text.
   - Keep paragraphs concise and easy to scan.

   HEADINGS:
   - Use Markdown headings (## or ###) only when the answer contains
     multiple distinct sections that genuinely need organization.
   - Do not use headings for simple, direct, or short answers.
   - Avoid generic headings such as "Introduction", "Overview",
     "Conclusion", or "Answer" unless genuinely necessary.

   EMPHASIS:
   - Use **bold** to highlight important concepts, keywords,
     constitutional articles, terms, names, or significant facts.
   - Use bold emphasis selectively and only where it improves clarity.
   - Do not bold entire paragraphs or overuse emphasis.

   LISTS:
   - Use unordered lists (-) for related points, characteristics,
     features, causes, effects, or multiple facts.
   - Use ordered lists (1.) for sequential steps, processes,
     procedures, or ranked items.
   - Keep list items concise and grammatically clear.
   - Do not convert every response into a list unnecessarily.

   COMPARISONS:
   - Use Markdown tables when multiple items need to be compared
     across common attributes.
   - Keep tables simple, concise, and relevant.
   - Avoid tables for simple explanations or when they reduce
     readability, especially on mobile screens.

   QUOTATIONS AND IMPORTANT INFORMATION:
   - Use Markdown blockquotes (>) only for meaningful quotations,
     definitions, or important statements that deserve emphasis.
   - Do not create artificial callout boxes using Markdown.
   - Clearly distinguish a factual statement from a quotation.

   MATHEMATICAL AND TECHNICAL CONTENT:
   - Format mathematical expressions appropriately when necessary.
   - Use inline code only for actual technical terms, commands,
     identifiers, or code.
   - Use fenced code blocks only when the student explicitly asks
     for code or when code is necessary to answer the question.

   FINAL FORMATTING GUIDELINES:
   - Formatting must improve clarity, not increase response length.
   - Do not force headings, bullets, tables, bold text, or other
     Markdown elements into every response.
   - Maintain a natural, educational, professional tone.
   - Ensure the final answer is clean, readable, and well-structured.
   - Return only the answer to the student without any additional
     commentary about how it was formatted.

Provided Context:
{context}
        """,
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


class _ExternalCitation(BaseModel):
    title: str = Field(
        description="Real name of the external publication, institution, or document. Never invent a name."
    )
    sourceUrl: str = Field(
        description=(
            "Required canonical http:// or https:// URL of this external source. "
            "Use the specific official or publisher page a student can open. "
            "Never leave this empty, and never return a source you cannot link."
        ),
    )


class _AnswerCoverage(BaseModel):
    answerCoveredByContext: bool = Field(
        description="True only when every material claim in the answer is supported by the retrieved Sumati IAS sources."
    )
    externalSources: list[_ExternalCitation] = Field(
        default_factory=list,
        description="Authorities used for claims the retrieved sources do not support. Empty when the context covers the answer.",
    )


_SOURCE_COVERAGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You compare a student answer with the Sumati IAS sources that were retrieved for it.

Decide whether those retrieved sources fully support the answer.
Treat rephrasing, summaries, and general wording of facts that appear
in the retrieved sources as covered.

Set answerCoveredByContext to false only when the answer states specific
facts, figures, names, dates, or explanations that are not present in
the retrieved sources.

When the retrieved sources do not cover the answer, list the external
authorities needed for the unsupported parts.
- Do not list a retrieved Sumati IAS source as an external source.
- Name only real publications, institutions, or official documents.
- Every external source must include sourceUrl. It is required, not optional.
- sourceUrl must be the real canonical http:// or https:// page for that
  source, specific enough for a student to open the document or article.
- If you cannot give both a real title and a real https URL, omit that
  source. Never return an external source with an empty or missing URL.
- If the retrieved sources fully support the answer, return no external sources.
- Return at most 5 external sources.
        """,
    ),
    (
        "human",
        """
Question:
{question}

Answer:
{answer}

Retrieved Sumati IAS sources:
{sources}
        """,
    ),
])




_STANDALONE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a question-rewriting assistant for Sumati IAS.

Your task is to rewrite the latest user question into a standalone,
contextually complete question using the conversation history.

STRICT RULES:

1. INTENT PRESERVATION:
   - Preserve the exact intent, meaning, and scope of the latest
     user question.
   - Do not change what the user is asking.
   - Do not make the question broader, narrower, or more specific
     than the original question.

2. CONTEXT RESOLUTION:
   - Use conversation history only to resolve references, missing
     context, entities, topics, or subjects.
   - Resolve pronouns and vague references such as "it", "this",
     "that", "they", "he", "she", "its", or "those" wherever
     their meaning is clear from the conversation history.
   - Include the relevant topic, entity, concept, or context in
     the rewritten question so it can be understood independently.
   - Do not unnecessarily repeat the entire conversation context.

3. ACCURACY:
   - Do not answer the question.
   - Do not add facts, assumptions, explanations, examples,
     or information not present in the original question
     or conversation history.
   - Do not infer missing information without sufficient context.
   - If the conversation history does not provide enough information
     to resolve a reference, preserve the original question rather
     than guessing.

4. QUESTION NORMALIZATION:
   - Rewrite the question using clear, natural, grammatically
     correct language.
   - Remove unnecessary conversational wording only when doing so
     does not change the user's intent.
   - Ensure the rewritten question is contextually complete and
     suitable for semantic search and information retrieval.
   - If the latest question is already standalone, return it
     unchanged except for minor grammatical improvements if necessary.

5. OUTPUT RULES:
   - Return exactly one standalone question.
   - Do not include introductory text, explanations, alternatives,
     answers, or reasoning.
   - Do not include labels such as "Rewritten question:".
   - Do not use Markdown formatting.
   - Do not wrap the output in quotation marks.
   - Return only the rewritten question.

"""
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


@traceable(name="standalone_question", run_type="llm")
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

    from server import signed_source_url

    messages = []
    signed_urls = {}
    for message in chat.get("messages") or []:
        created_at = message.get("createdAt")
        # print(message.get("sources"))
        sources = []
        for source in message.get("sources") or []:
            if not isinstance(source, dict):
                sources.append(source)
                continue

            if source.get("sourceDestination") == "external-source":
                source_url = (source.get("sourceUrl") or "").strip()
                if not source_url.lower().startswith(("http://", "https://")):
                    continue
                external = dict(source)
                external["sourceUrl"] = source_url
                external["sourceDestination"] = "external-source"
                sources.append(external)
                continue

            if source.get("resourceType") == "MAGAZINES":
                signed = dict(source)
                signed["sourceUrl"] = (f"/magazine-ebook?id={signed["documentId"]}")
                signed.setdefault("sourceDestination", "sumati-ias")
                sources.append(signed)
            else:
                signed = dict(source)
                signed["sourceUrl"] = signed_source_url(
                    signed.get("sourceUrl") or "",
                    signed_urls,
                )
                signed.setdefault("sourceDestination", "sumati-ias")
                sources.append(signed)

        messages.append({
            "role": message.get("role") or "",
            "content": message.get("content") or "",
            "sources": sources,
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
        "sourceDestination": "sumati-ias",
    }


def _retrieved_source_brief(documents):
    lines = []
    for index, document in enumerate(documents or [], start=1):
        metadata = document.metadata or {}
        title = metadata.get("title") or metadata.get("pageTitle") or ""
        source_url = metadata.get("sourceUrl") or ""
        text = " ".join((getattr(document, "page_content", "") or "").split())
        if len(text) > 700:
            text = text[:700] + "..."
        lines.append(
            f"[{index}] title={title}\nsourceUrl={source_url}\ntext={text}"
        )
    return "\n\n".join(lines) if lines else "(none)"


def _citation_field(citation, key):
    if isinstance(citation, dict):
        return citation.get(key) or ""
    return getattr(citation, key, "") or ""


def _external_source(citation):
    title = str(_citation_field(citation, "title") or "").strip()
    source_url = str(_citation_field(citation, "sourceUrl") or "").strip()
    if not source_url.lower().startswith(("http://", "https://")):
        return None
    return {
        "documentId": "",
        "resourceType": "",
        "sourceUrl": source_url,
        "title": title or source_url,
        "chunkIndex": None,
        "sourceDestination": "external-source",
    }


def _matches_listed_source(candidate, sources):
    title = (candidate.get("title") or "").strip().lower()
    source_url = (candidate.get("sourceUrl") or "").strip().lower()
    for source in sources:
        if source_url and (source.get("sourceUrl") or "").strip().lower() == source_url:
            return True
        if title and (source.get("title") or "").strip().lower() == title:
            return True
    return False


@traceable(name="classify_web_sources", run_type="chain", tags=["web"])
def _external_citations(question, answer, documents):
    """Name outside authorities only when the dataset does not cover the answer."""
    if not (answer or "").strip():
        return []

    try:
        coverage = (
            _SOURCE_COVERAGE_PROMPT
            | ChatOpenAI(model=LLM_MODEL, temperature=0).with_structured_output(
                _AnswerCoverage
            )
        ).invoke({
            "question": question,
            "answer": answer,
            "sources": _retrieved_source_brief(documents),
        })

    except Exception:
        return []

    if isinstance(coverage, dict):
        covered = coverage.get("answerCoveredByContext")
        citations = coverage.get("externalSources") or []
    else:
        covered = getattr(coverage, "answerCoveredByContext", True)
        citations = getattr(coverage, "externalSources", None) or []

    if covered:
        return []
    return list(citations)[:5]


@traceable(name="answer_web_question", run_type="chain", tags=["web"])
def answer_web_question(question, chat_history=None):
    """Answer from all scraped web chunks, using this browser's saved chat only."""

    history = _history_messages(chat_history or [])
    search_question = _standalone_question(question, history)

    retriever = get_public_data_vector_store().as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 1,
            "score_threshold": 0.82 # relevancy score threshold for finding documents
        }
    )

    # results = get_public_data_vector_store().similarity_search_with_relevance_scores("INS Betwa", k=4)

    # for doc, score in results:
    #     print(f"Score: {score:.4f}")
    #     print(f"Source: {doc.metadata.get('sourceUrl')}")
    #     print(f"Content: {doc.page_content[:300]}")
    #     print("-" * 80)

    @traceable(name="retrieve_web_chunks", run_type="retriever")
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

    print(result)

    answer = result.get("answer") or ""

    if not isinstance(answer, str):
        answer = getattr(answer, "content", str(answer))

    sources = []
    context_documents = result.get("context") or []

    for document in context_documents:
        source = _web_source(document)
        if not any(existing["documentId"] == source["documentId"] for existing in sources):
            sources.append(source)

    for citation in _external_citations(question, answer, context_documents):
        external = _external_source(citation)
        if external is None or _matches_listed_source(external, sources):
            continue
        sources.append(external)

    return {
        "answer": answer,
        "sources": sources,
    }


def _lecture_question_inputs(inputs):
    return {"question": inputs.get("question")}


@traceable(
    name="answer_lecture_question",
    run_type="chain",
    tags=["lecture"],
    process_inputs=_lecture_question_inputs,
)
def answer_lecture_question(question, chain):
    return chain.invoke({"question": question})


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