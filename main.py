
from langchain_docling.loader import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_openai import ChatOpenAI
# from langchain.memory import ConversationBufferMemory

FILE_PATH = "./assets/L2_Making_of_the_Indian_Constitution.docx"
OUTPUT_FILE = "./assets/extracted_content.txt"

loader = DoclingLoader(file_path=FILE_PATH)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)

print(f"docs {docs}")

# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small"
# )

# vectorstore = Chroma.from_documents(
#     docs,
#     embeddings,
#     persist_directory="vectordb"
# )

# retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0
# )

# memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     return_messages=True
# )

with open("./doc-temp.txt", "a", encoding="utf-8") as f:
    f.write(docs)

# Append extracted content to the text file
with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
    for document in documents:
        file.write(document.page_content)
        file.write("\n\n" + "=" * 80 + "\n\n")

print(f"Successfully appended {len(documents)} documents to {OUTPUT_FILE}")
