# Simple RAG Demo

## Setup
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Add your OpenAI API key to the `.env` file.

## Execution
1. Run ingestion to create the vector database:
   `python src/ingest.py`
2. Run the main script to query the RAG system:
   `python src/main.py`
