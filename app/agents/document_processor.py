from langchain.text_splitter import RecursiveCharacterTextSplitter
from termcolor import colored
from typing import List, Dict
import os
from .vector_store import VectorStore
from .knowledge_graph import KnowledgeGraph
import asyncio

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vector_store = VectorStore()
        self.knowledge_graph = KnowledgeGraph()
        self._kg_lock = asyncio.Lock()  # Lock for synchronizing knowledge graph updates
    
    async def extract_text(self, file_path: str) -> str:
        """Extract text from uploaded document."""
        try:
            print(colored(f"Extracting text from {file_path}", "cyan"))
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            return text
        except Exception as e:
            print(colored(f"Error extracting text: {str(e)}", "red"))
            raise

    async def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for processing."""
        try:
            print(colored("Chunking text...", "cyan"))
            chunks = self.text_splitter.split_text(text)
            print(colored(f"Created {len(chunks)} chunks", "green"))
            return chunks
        except Exception as e:
            print(colored(f"Error chunking text: {str(e)}", "red"))
            raise

    async def process_document(self, file_path: str) -> Dict:
        """Process document through extraction, chunking, vector store, and knowledge graph."""
        try:
            # Extract and chunk text
            text = await self.extract_text(file_path)
            chunks = await self.chunk_text(text)
            
            # Add chunks to vector store
            metadata = {"source": file_path}
            await self.vector_store.add_chunks(chunks, metadata)
            
            # Update knowledge graph with lock to prevent concurrent updates
            async with self._kg_lock:
                print(colored(f"[PROCESSOR] Acquired lock for knowledge graph update - {file_path}", "cyan"))
                try:
                    graph_stats = await self.knowledge_graph.update_graph(chunks)
                finally:
                    print(colored(f"[PROCESSOR] Released lock for knowledge graph update - {file_path}", "cyan"))
            
            return {
                "text": text,
                "chunks": chunks,
                "num_chunks": len(chunks),
                "graph_stats": graph_stats
            }
        except Exception as e:
            print(colored(f"Error processing document: {str(e)}", "red"))
            raise 