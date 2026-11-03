import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from rag_chain import get_vector_store, collection
from bson import ObjectId


def ingest_pdf(file_path, lecture_id):
    """Load a PDF into its own vector database."""
    if not os.path.isfile(file_path):
        raise ValueError(f"PDF not found: {file_path}")

    if not lecture_id or not lecture_id.strip():
        raise ValueError("lecture_id is required")

    lecture_id = lecture_id.strip()

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    pages_with_text = [doc for doc in documents if doc.page_content.strip()]

    if not pages_with_text:
        raise ValueError("PDF has no extractable text")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = text_splitter.split_documents(pages_with_text)

    if not docs:
        raise ValueError("PDF produced no text chunks")

    for index, doc in enumerate(docs):
        doc.metadata.clear()
        doc.metadata.update({
            "lectureId": ObjectId(lecture_id),
            "chunkIndex": index,
        })

    vectorstore = get_vector_store()

    # 5. Remove previously indexed chunks for this lecture
    collection.delete_many({
        "lectureId": lecture_id
    })

    # 6. Generate embeddings and insert into MongoDB
    inserted_ids = vectorstore.add_documents(
        documents=docs
    )

    return {
        "filename": os.path.basename(file_path),
        "pages": len(documents),
        "chunks": len(docs),
        "inserted": len(inserted_ids),
        "lectureId": lecture_id,
    }


if __name__ == "__main__":
    summary = ingest_pdf(
        file_path="data/sample_rag_test.pdf",
        lecture_id="test_lecture_123",
    )