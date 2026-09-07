from dotenv import load_dotenv
from retriever import get_retriever
from generator import create_rag_chain

def main():
    # Load environment variables (API Key)
    load_dotenv()

    print("Initializing RAG components...")
    retriever = get_retriever()
    rag_chain = create_rag_chain(retriever)

    # Ask a question about the custom data
    question = "What is the battery life of the Quantum Gizmo V2 and is it waterproof?"
    print(f"\nQuestion: {question}")
    
    print("Generating answer...")
    response = rag_chain.invoke(question)
    
    print(f"\nAnswer: {response}")

if __name__ == "__main__":
    main()
