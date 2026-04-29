from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
import shutil

NEW_DIR = "data/raw/new"
RAW_DIR = "data/raw"

# embeds PDFs and saves the vectors to disk, later searchable via ChromaDB cosine similarity
def ingest_sources():
    new_files = [f for f in os.listdir(NEW_DIR) if f.endswith(".pdf")]

    if not new_files:
        print("No new files to ingest")
        return

    # chunk documents due to LLM context window limits
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200  # overlap ensures full representation in at least one chunk
    )

    chunks = []
    for filename in new_files:
        docs = PyPDFLoader(f"{NEW_DIR}/{filename}").load()
        chunks.extend(splitter.split_documents(docs))
    print(f"Created {len(chunks)} chunks from {len(new_files)} new documents")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # semantic search handles synonyms, paraphrasing, and conceptual similarity
    if os.path.exists("data/chroma_db"):
        # add to existing vectorstore to avoid re-embedding previously ingested files
        vectorstore = Chroma(
            persist_directory="data/chroma_db",
            embedding_function=embeddings
        )
        vectorstore.add_documents(chunks)
    else:
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="data/chroma_db"
        )

    # move ingested files from new/ to raw/
    for filename in new_files:
        shutil.move(f"{NEW_DIR}/{filename}", f"{RAW_DIR}/{filename}")
    print(f"Vector store updated — {len(new_files)} new files ingested")

# convert list of Document objects from ChromaDB into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)