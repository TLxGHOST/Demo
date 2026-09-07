import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def ingest_data():
    # 1. Load the document
    loader = TextLoader("data/sample.txt")
    docs = loader.load()

    # 2. Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = text_splitter.split_documents(docs)

    # 3. Create embeddings and store them in a local Chroma vector database
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory="./chroma_db"
    )
    print(f"Successfully ingested {len(splits)} chunks into ChromaDB.")

if __name__ == "__main__":
    ingest_data()
