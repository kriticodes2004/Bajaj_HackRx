import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Path to your data folder
pdf_folder = "data"

# Load all PDFs
docs = []
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        path = os.path.join(pdf_folder, filename)
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

print(f"📄 Loaded {len(docs)} PDF documents.")

# Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"🔪 Chunked into {len(chunks)} segments.")

# Embedding model
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Chroma vector store
persist_directory = "chromadb"
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=persist_directory
)
vectordb.persist()
print(f"✅ Done. {len(chunks)} chunks embedded and saved to ChromaDB.")
