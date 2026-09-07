from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def get_retriever():
    # Load the existing database from the persist directory
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=OpenAIEmbeddings()
    )
    # Return a retriever that fetches the top 2 most relevant chunks
    return vectorstore.as_retriever(search_kwargs={"k": 2})
