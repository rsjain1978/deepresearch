# Deep Research Application

An intelligent document processing and search application that combines document extraction, knowledge graph creation, and semantic search capabilities.

## Features

- Document upload and text extraction
- Automatic knowledge graph creation
- Advanced semantic search with multiple search terms
- Vector database integration
- AI-powered answer generation

## Setup

1. Create and activate the langgraph environment:
```bash
conda create -n langgraph python=3.11
conda activate langgraph
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env` file:
- OPENAI_API_KEY
- TAVILY_API_KEY

4. Run the application:
```bash
uvicorn app.main:app --reload
```

The application will be available at http://localhost:8000 