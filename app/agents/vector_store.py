from chromadb import Client, Settings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from termcolor import colored
import os
from typing import List, Dict, Any
import asyncio
import shutil

class VectorStore:
    def __init__(self):
        print(colored("[VECTOR] Initializing Vector Store with ChromaDB", "cyan"))
        
        # Initialize the language model
        self.llm = ChatOpenAI(model_name="gpt-4o")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Set up ChromaDB client
        self.persist_directory = "./data/chroma"
        
        # Initialize Chroma through LangChain
        self.vectorstore = Chroma(
            collection_name="documents",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Create the retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": 3,
            }
        )
        
        # Initialize the ConversationalRetrievalChain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,  # Include source documents in response
            verbose=True
        )
        
        self._lock = asyncio.Lock()  # Lock for synchronizing vector store updates
        print(colored(f"[VECTOR] Initialized LangChain Chroma vectorstore with {self.vectorstore._collection.count()} documents", "cyan"))

    async def add_chunks(self, chunks: List[str], metadata: Dict = None) -> None:
        """Add document chunks to the vector store."""
        try:
            print(colored(f"[VECTOR] Adding {len(chunks)} chunks to vector store...", "cyan"))
            
            async with self._lock:
                # Add documents to Chroma
                print(colored(f"[VECTOR] Adding documents to Chroma with metadata: {metadata}", "cyan"))
                
                # Prepare metadata for each chunk
                metadatas = [{**metadata, "chunk_id": i} for i in range(len(chunks))]
                
                # Add documents to vectorstore
                self.vectorstore.add_texts(
                    texts=chunks,
                    metadatas=metadatas
                )
                
                # Persist the changes
                self.vectorstore.persist()
                
                print(colored(f"[VECTOR] Successfully added {len(chunks)} chunks to vector store", "cyan"))
        except Exception as e:
            print(colored(f"[VECTOR] Error adding chunks to vector store: {str(e)}", "red"))
            raise

    async def search(
        self,
        query: str,
        n_results: int = 3  # Align with retriever's k value
    ) -> List[Dict]:
        """Search for similar chunks using LangChain's ConversationalRetrievalChain."""
        try:
            print(colored(f"\n[VECTOR] Starting search for query: '{query}'", "cyan"))
            doc_count = self.vectorstore._collection.count()
            print(colored(f"[VECTOR] Total documents in collection: {doc_count}", "cyan"))
            
            if doc_count == 0:
                print(colored("[VECTOR] No documents in collection, returning empty results", "yellow"))
                return []

            print(colored("[VECTOR] Performing conversational search...", "cyan"))
            result = await self.qa_chain.acall({
                "question": query,
                "chat_history": []
            })

            # Process source documents
            matches = []
            source_docs = result.get("source_documents", [])
            print(colored(f"\n[VECTOR] Found {len(source_docs)} source documents", "cyan"))

            for doc in source_docs:
                print(colored(f"\n[VECTOR] Processing document:", "cyan"))
                print(colored(f"- Metadata: {doc.metadata}", "cyan"))
                print(colored(f"- Text snippet: {doc.page_content[:200]}...", "cyan"))
                
                matches.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "answer": result["answer"]
                })

            print(colored(f"\n[VECTOR] Returning {len(matches)} matches", "cyan"))
            return matches

        except Exception as e:
            print(colored(f"[VECTOR] Error during search: {str(e)}", "red"))
            raise

    def clear(self) -> None:
        """Delete and recreate the collection and clean up persistent storage."""
        try:
            print(colored("[VECTOR] Deleting vector store collection", "cyan"))
            
            self.vectorstore.delete_collection()
            
            print(colored("[VECTOR] Vector store cleared and reinitialized successfully", "cyan"))
        except Exception as e:
            print(colored(f"[VECTOR] Error clearing vector store: {str(e)}", "red"))
            raise 