from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ..agents.pdf_processor import DocumentProcessor
from ..agents.search import SearchAgent
from termcolor import colored
import os
import shutil
from typing import Dict, List
from pydantic import BaseModel
from ..agents.knowledge_graph import KnowledgeGraph

router = APIRouter()
document_processor = DocumentProcessor()
search_agent = SearchAgent(document_processor.vector_store, document_processor.knowledge_graph)

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

class SearchQuery(BaseModel):
    query: str
    num_results: int = 5
    use_kg: bool = False  # Default to not using knowledge graph

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict:
    """
    Upload and process a document.
    """
    try:
        print(colored(f"[API] Receiving file: {file.filename}", "green"))
        
        # Save the uploaded file
        file_path = os.path.join("uploads", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(colored(f"[API] File saved at: {file_path}", "green"))
        
        # Process the document
        result = await document_processor.process_document(file_path)
        
        # Return clean response without debug logs
        return {
            "status": "success",
            "message": "Document processed successfully",
            "filename": file.filename,
            "num_pages": result["num_pages"],
            "num_chunks": result["num_chunks"],
            "graph_stats": {
                "num_nodes": result["graph_stats"]["num_nodes"],
                "num_edges": result["graph_stats"]["num_edges"],
                "node_types": result["graph_stats"]["node_types"],
                "relation_types": result["graph_stats"]["relation_types"]
            }
        }
    
    except Exception as e:
        error_msg = str(e)
        print(colored(f"[API] Error processing upload: {error_msg}", "red"))
        raise HTTPException(status_code=500, detail="Error processing document. Check server logs for details.")

@router.get("/knowledge-graph")
async def get_knowledge_graph() -> Dict:
    """
    Get the current state of the knowledge graph for visualization.
    """
    try:
        print(colored("[API] Retrieving knowledge graph data for visualization", "green"))
        graph_data = document_processor.knowledge_graph.get_graph_data()
        
        # Log summary info to console only
        print(colored(f"[API] Returning graph with {len(graph_data['nodes'])} nodes and {len(graph_data['links'])} relationships", "green"))
        
        return graph_data
    except Exception as e:
        error_msg = str(e)
        print(colored(f"[API] Error getting knowledge graph: {error_msg}", "red"))
        raise HTTPException(status_code=500, detail="Error retrieving knowledge graph. Check server logs for details.")

@router.post("/search")
async def search(query: SearchQuery) -> Dict:
    """
    Search through documents using advanced search with multiple terms and re-ranking.
    If use_kg is True, also includes knowledge graph in search.
    """
    try:
        print(colored(f"[API] Processing search query: '{query.query}' with n_results={query.num_results}, use_kg={query.use_kg}", "green"))
        
        # Perform advanced search
        search_results = await search_agent.search(query.query, use_kg=query.use_kg)
        
        print(colored(f"[API] Search completed with {len(search_results['results'])} results", "green"))
        
        # Return clean response without debug logs
        return {
            "status": "success",
            "query": search_results["query"],
            "similar_queries": search_results["similar_queries"],
            "results": search_results["results"],
            "answer": search_results["answer"]
        }
    except Exception as e:
        error_msg = str(e)
        print(colored(f"[API] Error performing search: {error_msg}", "red"))
        raise HTTPException(status_code=500, detail="Error performing search. Check server logs for details.")

@router.delete("/knowledge-graph")
async def delete_knowledge_graph() -> Dict:
    """
    Delete the current knowledge graph and vector store data.
    """
    try:
        print(colored("[API] Deleting knowledge graph and vector store", "green"))
        
        # Clear vector store
        document_processor.vector_store.clear()
        
        # Clear knowledge graph
        document_processor.knowledge_graph.clear()
        
        return {
            "status": "success",
            "message": "Knowledge base deleted successfully"
        }
    except Exception as e:
        error_msg = str(e)
        print(colored(f"[API] Error deleting knowledge base: {error_msg}", "red"))
        raise HTTPException(status_code=500, detail="Error deleting knowledge base. Check server logs for details.") 