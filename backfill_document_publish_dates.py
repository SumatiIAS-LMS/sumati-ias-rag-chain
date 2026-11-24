# """Temporary helper: copy updatedAt onto chunks as documentPublishDate.

# Run once with the project virtualenv:

#     .venv\\Scripts\\python.exe backfill_document_publish_dates.py
# """

# from bson import ObjectId

# from config import (
#     MONGODB_COLLECTION,
#     MONGODB_DATA_CHUNKS_COLLECTION,
#     MONGODB_LECTURES_COLLECTION,
#     MONGODB_WEB_CURRENT_AFFAIRS_COLLECTION,
#     MONGODB_WEB_EDITORIAL_COLLECTION,
#     MONGODB_WEB_HOT_TOPIC_COLLECTION,
# )
# from rag_chain import database, document_publish_date

# # resourceType values used when a chunk has no sourceCollection.
# _RESOURCE_COLLECTIONS = {
#     "current-affair": MONGODB_WEB_CURRENT_AFFAIRS_COLLECTION,
#     "web-hot-topic": MONGODB_WEB_HOT_TOPIC_COLLECTION,
#     "web-editorial": MONGODB_WEB_EDITORIAL_COLLECTION,
# }


# def _source_id(value):
#     if isinstance(value, ObjectId):
#         return value
#     if isinstance(value, str) and ObjectId.is_valid(value):
#         return ObjectId(value)
#     return None


# def _source_collection_name(source_collection, resource_type):
#     name = (source_collection or "").strip()
#     if name:
#         return name
#     return _RESOURCE_COLLECTIONS.get((resource_type or "").strip())


# def _updated_at(collection_name, document_id):
#     source_id = _source_id(document_id)
#     if collection_name is None or source_id is None:
#         return None
#     source = database[collection_name].find_one(
#         {"_id": source_id},
#         {"updatedAt": 1},
#     )
#     if source is None:
#         return None
#     return document_publish_date(source.get("updatedAt"))


# def _backfill_web_chunks():
#     chunks = database[MONGODB_DATA_CHUNKS_COLLECTION]
#     groups = chunks.aggregate([
#         {
#             "$group": {
#                 "_id": {
#                     "documentId": "$documentId",
#                     "sourceCollection": "$sourceCollection",
#                     "resourceType": "$resourceType",
#                 }
#             }
#         }
#     ])

#     updated_documents = 0
#     updated_chunks = 0
#     skipped = 0

#     for group in groups:
#         key = group["_id"]
#         document_id = key.get("documentId")
#         collection_name = _source_collection_name(
#             key.get("sourceCollection"),
#             key.get("resourceType"),
#         )
#         published = _updated_at(collection_name, document_id)
#         if published is None:
#             skipped += 1
#             print(
#                 f"[SKIP] {MONGODB_DATA_CHUNKS_COLLECTION} "
#                 f"documentId={document_id} collection={collection_name}"
#             )
#             continue

#         query = {"documentId": document_id}
#         if key.get("sourceCollection"):
#             query["sourceCollection"] = key.get("sourceCollection")
#         if key.get("resourceType"):
#             query["resourceType"] = key.get("resourceType")

#         result = chunks.update_many(
#             query,
#             {"$set": {"documentPublishDate": published}},
#         )
#         updated_documents += 1
#         updated_chunks += result.modified_count

#     return updated_documents, updated_chunks, skipped


# def _backfill_lecture_chunks():
#     chunks = database[MONGODB_COLLECTION]
#     lecture_ids = chunks.distinct("lectureId")

#     updated_documents = 0
#     updated_chunks = 0
#     skipped = 0

#     for lecture_id in lecture_ids:
#         published = _updated_at(MONGODB_LECTURES_COLLECTION, lecture_id)
#         if published is None:
#             skipped += 1
#             print(
#                 f"[SKIP] {MONGODB_COLLECTION} "
#                 f"documentId={lecture_id} collection={MONGODB_LECTURES_COLLECTION}"
#             )
#             continue

#         result = chunks.update_many(
#             {"lectureId": lecture_id},
#             {"$set": {"documentPublishDate": published}},
#         )
#         updated_documents += 1
#         updated_chunks += result.modified_count

#     return updated_documents, updated_chunks, skipped


# def backfill_document_publish_dates():
#     """Set documentPublishDate on existing web, PDF, and lecture chunks."""
#     web_documents, web_chunks, web_skipped = _backfill_web_chunks()
#     lecture_documents, lecture_chunks, lecture_skipped = _backfill_lecture_chunks()
#     total_documents = web_documents + lecture_documents

#     print(f"{MONGODB_DATA_CHUNKS_COLLECTION}: {web_documents} documents, {web_chunks} chunks")
#     print(f"{MONGODB_COLLECTION}: {lecture_documents} documents, {lecture_chunks} chunks")
#     print(f"Skipped documents: {web_skipped + lecture_skipped}")
#     print(f"Total documents updated: {total_documents}")
#     return total_documents


# if __name__ == "__main__":
#     backfill_document_publish_dates()
