import asyncio
import hashlib
import json
from typing import List, Dict

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from termcolor import colored
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import spacy

from .base import call_llm
from .vector_store import VectorStore
from .knowledge_graph import KnowledgeGraph

# Load NLP and embedding models
spacy_nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()

def deduplicate_queries(queries: List[str], threshold: float = 0.9) -> List[str]:
    embeddings = embedding_model.encode(queries, convert_to_tensor=True)
    keep = []
    for i, emb in enumerate(embeddings):
        if all(util.cos_sim(emb, e)[0][0] < threshold for e in embedding_model.encode(keep, convert_to_tensor=True)):
            keep.append(queries[i])
    return keep

def normalize_text(text):
    if isinstance(text, list):
        return ' '.join(map(str, text))
    return str(text)

class SearchAgent:
    def __init__(self, vector_store: VectorStore, knowledge_graph: KnowledgeGraph):
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        print(colored("[SEARCH] Simplified SearchAgent Initialized", "blue"))

    async def generate_similar_queries(self, query: str) -> List[str]:
        """Generate 3 alternative search queries, not including the original query."""
        print(colored(f"\n[SEARCH] Generating queries for each entity in the query: '{query}'", "blue"))
        
        try:
            prompt_template = PromptTemplate(
                input_variables=["query"],
                template="""
            You are a query rewriter. Break down the user's query into individual sub-queries for each entity mentioned.

            Original Query: "{query}"

            Rewritten Queries:
            1.
            """
            )            

            llm = ChatOpenAI(model_name="gpt-4o", 
                             temperature=0)  # You can also use ChatOpenAI or other LLMs

            query_rewriter_chain = LLMChain(
                llm=llm,
                prompt=prompt_template,
                verbose=True
            )

            rewritten_output = query_rewriter_chain.run(query=query)

            # Parse the output
            subqueries = [line.strip("0123456789. ") for line in rewritten_output.strip().split("\n") if line.strip()]


            print(colored("\n[SEARCH] Alternative search queries:", "blue"))
            for i, q in enumerate(subqueries):
                print(colored(f"{i+1}. {q}", "blue"))
            
            return subqueries
            
        except Exception as e:
            print(colored(f"[SEARCH] Error generating alternative queries: {str(e)}", "red"))
            return [
                f"details of {query}",
                f"information about {query}",
                f"data regarding {query}"
            ]

    async def search_vector_store(self, queries: List[str]) -> List[Dict]:
        """Search vector store with multiple queries and deduplicate results."""
        print(colored(f"[SEARCH] Searching vector store with queries: {queries}", "blue"))

        seen_hashes = set()
        all_results = []

        # Create search tasks for each query
        search_tasks = [self.vector_store.search(q) for q in queries]
        search_results = await asyncio.gather(*search_tasks)

        # Process results from all queries
        for query, results in zip(queries, search_results):
            for result in results:
                # Hash the text content for deduplication
                text = result.get("text", "").strip()
                h = hash_text(text)
                
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    all_results.append({
                        "text": text,
                        "source_type": "vector_db",
                        "metadata": result.get("metadata", {}),
                        "answer": result.get("answer", "")
                    })
                    print(colored(f"[SEARCH] Added result from query: '{query}'", "blue"))

        print(colored(f"[SEARCH] Found {len(all_results)} unique results", "blue"))
        return all_results

    async def search_knowledge_graph(self, query: str) -> List[Dict]:
        results = []
        try:
            kg_results = await self.knowledge_graph.search_graph(query)
            for item in kg_results.get('results', []):
                results.append({
                    'text': item.get('content', ''),
                    'source_type': 'knowledge_graph',
                    'relevance_score': item.get('score', 0.5),
                    'metadata': item.get('metadata', {})
                })
        except Exception as e:
            print(colored(f"[SEARCH] KG Search failed: {str(e)}", "red"))
        return results

    def rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        pairs = [[query, r['text']] for r in results]
        scores = cross_encoder.predict(pairs)
        
        # Normalize scores to be between 0 and 1
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        for r, s in zip(results, scores):
            # Normalize score if we have a valid range, otherwise default to 0.5
            if score_range > 0:
                normalized_score = (s - min_score) / score_range
            else:
                normalized_score = 0.5
            r['relevance_score'] = float(normalized_score)
        return results

    def apply_source_confidence(self, results: List[Dict]) -> List[Dict]:
        weights = {
            'vector_db': 1.0,
            'knowledge_graph': 1.1
        }
        for r in results:
            base = r.get('relevance_score', 0.5)
            weight = weights.get(r.get('source_type', 'vector_db'), 1.0)
            r['adjusted_score'] = base * weight
        return results

    async def generate_answer(self, query: str, results: List[Dict]) -> str:
        try:
            prompt = {
                "role": "system",
                "content": """You're a precise assistant. Generate a markdown answer strictly using the search results. Do not add extra knowledge. Mention conflicting or missing info transparently.\nFormat: headings, subheadings, bullets, blockquotes, citations."""
            }
            context = "\n\n".join([
                f"Source {i+1} ({r['source_type']}):\n{r['text']}" for i, r in enumerate(results)
            ])
            user_prompt = {
                "role": "user",
                "content": f"Query: {query}\n\nSearch Results:\n{context}"
            }
            return await call_llm([prompt, user_prompt])
        except Exception as e:
            print(colored(f"[SEARCH] Answer generation error: {str(e)}", "red"))
            return "Unable to generate a comprehensive answer at this time."

    async def search(self, query: str, use_kg: bool = True) -> Dict:
        print(colored(f"\n[SEARCH] Starting search process for query: '{query}'", "blue"))
        print(colored(f"[SEARCH] Knowledge graph enabled: {use_kg}", "blue"))
        
        # Step 1: Generate similar queries
        similar_queries = await self.generate_similar_queries(query)
        
        # Step 2: Search vector store
        print(colored("\n[SEARCH] Searching vector store...", "blue"))
        vector_results = await self.search_vector_store(similar_queries)
        print(colored(f"[SEARCH] Vector store returned {len(vector_results)} results", "blue"))
        
        # Step 3: Search knowledge graph if enabled
        kg_results = []
        if use_kg:
            print(colored("\n[SEARCH] Searching knowledge graph...", "blue"))
            kg_search_tasks = [self.search_knowledge_graph(q) for q in similar_queries]
            kg_results_list = await asyncio.gather(*kg_search_tasks)
            
            # Deduplicate knowledge graph results
            seen_content = set()
            for results in kg_results_list:
                for result in results:
                    content = result.get('text', '').strip()
                    if content and content not in seen_content:
                        seen_content.add(content)
                        kg_results.append(result)
            print(colored(f"[SEARCH] Knowledge graph returned {len(kg_results)} unique results", "blue"))

        # Step 4: Combine and process results
        all_results = vector_results + kg_results
        print(colored(f"\n[SEARCH] Total results before ranking: {len(all_results)}", "blue"))
        
        if not all_results:
            print(colored("[SEARCH] No results found", "yellow"))
            return {
                "query": query,
                "similar_queries": similar_queries,
                "results": [],
                "answer": "No relevant results found."
            }

        # Step 5: Rerank results
        print(colored("\n[SEARCH] Reranking results...", "blue"))
        ranked = self.rerank_results(query, all_results)
        ranked = self.apply_source_confidence(ranked)
        ranked = sorted(ranked, key=lambda x: x['adjusted_score'], reverse=True)
        print(colored(f"[SEARCH] Reranking complete. Taking top 5 results.", "blue"))

        # Step 6: Generate answer
        print(colored("\n[SEARCH] Generating answer from top results...", "blue"))
        answer = await self.generate_answer(query, ranked[:5])

        return {
            "query": query,
            "similar_queries": similar_queries,
            "results": ranked[:5],
            "answer": answer
        }
