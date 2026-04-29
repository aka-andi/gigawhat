# "gigaWhat?" energy query engine (WIP)
a RAG pipeline over public [EIA](https://www.eia.gov/) energy reports, enabling natural language querying with source-attributed responses

## how it works
PDF reports are chunked and embedded using OpenAI's `text-embedding-3-small` model, then stored in a local ChromaDB vector store. at query time, the question is embedded and the most semantically similar chunks are retrieved via cosine similarity. the retrieved context is passed alongside the question to GPT-4o-mini via a LangChain chain, which generates a grounded, source-cited response.

## setup
1. clone the repo and create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # windows: venv\Scripts\activate
```
2. install dependencies
```bash
pip install langchain langchain-openai langchain-chroma langchain-text-splitters langchain-community chromadb ragas fastapi uvicorn python-dotenv pypdf
```
3. add your OpenAI API key to a `.env` file
```
OPENAI_API_KEY=your_key_here
```
4. download EIA energy reports (PDF) from [eia.gov](https://www.eia.gov/reports/) and place them in `data/raw/new/`

## usage
```bash
python main.py
```
on first run, files in `data/raw/new/` are ingested and moved to `data/raw/`. subsequent runs skip ingestion unless new files are present in `data/raw/new/`.

## project structure
```
gigawhat/
├── main.py          # query chain and response loop
├── configs.py       # configuration constants
├── utils.py         # PDF loading, chunking, embedding, persistence, helpers
├── data/
│   ├── raw/         # ingested PDFs
│   │   └── new/     # path to add new PDFs for ingestion
│   └── chroma_db/   # persisted vector store (not tracked)
```