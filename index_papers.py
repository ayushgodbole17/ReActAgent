# index_papers.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load your OpenAI API key from .env
load_dotenv()

# 1. Initialize the embedding model
embeddings = OpenAIEmbeddings()

# 2. Load PDFs from the data/ folder
loaders = [
    PyPDFLoader(os.path.join("data", f))
    for f in os.listdir("data")
    if f.lower().endswith(".pdf")
]

# 3. Split each document into chunks (~1000 characters, 200 overlap)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = []
for loader in loaders:
    docs.extend(loader.load_and_split(splitter))

# 4. Build (or load) the Chroma vector store and persist it
store = Chroma.from_documents(docs, embeddings, persist_directory="vector_db")
store.persist()

print(f"Indexed {len(docs)} document chunks.")
