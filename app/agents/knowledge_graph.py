from .base import call_llm
from termcolor import colored
from typing import List, Dict
import json
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
import asyncio
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_neo4j import Neo4jVector, Neo4jGraph, GraphCypherQAChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import hashlib

# Load environment variables
load_dotenv()

class KnowledgeGraph:
    def __init__(self):
        # Get Neo4j connection details from environment variables
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "research_dbms")
        self.database = os.getenv("NEO4J_DATABASE", "research_dbms")
        
        # Initialize components
        self._driver = None
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.graph = None
        self.qa_chain = None
        
        # Connect and initialize
        self._connect()
        self._init_db()
        self._init_search()
            
    def _connect(self):
        """Connect to Neo4j database."""
        try:
            print(colored(f"[NEO4J] Connecting to Neo4j at {self.uri}...", "magenta"))
            self._driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Test the connection
            with self._driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS result")
                record = result.single()
                assert record["result"] == 1
            print(colored("[NEO4J] Connected to Neo4j successfully.", "magenta"))
        except Exception as e:
            print(colored(f"[NEO4J] Error connecting to Neo4j: {str(e)}", "red"))
            raise
            
    def _init_db(self):
        """Initialize database with constraints if needed."""
        if not self._driver:
            raise Exception("[NEO4J] No database connection available")
            
        try:
            with self._driver.session(database=self.database) as session:
                # Create constraints
                constraints = [
                    """CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
                       FOR (e:Entity) REQUIRE e.id IS UNIQUE""",
                    """CREATE CONSTRAINT text_chunk_id_unique IF NOT EXISTS
                       FOR (c:TextChunk) REQUIRE c.chunk_id IS UNIQUE"""
                ]
                
                for constraint in constraints:
                    session.run(constraint)
                    
                print(colored("[NEO4J] Database initialized with constraints.", "magenta"))
        except Exception as e:
            print(colored(f"[NEO4J] Error initializing database: {str(e)}", "red"))
            raise

    def _init_search(self):
        """Initialize the search components using LangChain."""
        if not self._driver:
            raise Exception("[NEO4J] No database connection available")
            
        try:
            # Initialize Neo4j components
            self.graph = Neo4jGraph(
                url=self.uri,
                username=self.username,
                password=self.password,
                database=self.database
            )
            
            # Custom query template to avoid duplicate columns
            CYPHER_TEMPLATE = """
            You are a Cypher expert. Given an input question, create a Cypher query that will answer the question.
            Be careful to not return duplicate column names in your query.
            Return only the Cypher query without any explanation.

            For example:
            Question: What entities are related to Tesla?
            Answer: MATCH (e1:Entity {name: "TESLA"})-[r:RELATES]->(e2:Entity) RETURN e2.name as name, e2.type as type, e2.id as id, r.type as relationship

            Question: {question}
            Answer: """

            self.qa_chain = GraphCypherQAChain.from_llm(
                ChatOpenAI(temperature=0),
                graph=self.graph,
                verbose=True,
                cypher_prompt_template=CYPHER_TEMPLATE,
                allow_dangerous_requests=True
            )
            
            print(colored("[KG] Initialized Neo4j search components", "magenta"))
        except Exception as e:
            print(colored(f"[KG] Error initializing search components: {str(e)}", "red"))
            raise

    def close(self):
        """Close the Neo4j connection."""
        if self._driver:
            print(colored("[NEO4J] Closing Neo4j connection.", "magenta"))
            self._driver.close()

    def extract_triples_from_text(self, text: str) -> List[Dict[str, str]]:
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate

        prompt = PromptTemplate.from_template("""
        Extract factual triples (subject, predicate, object) from the following text:

        {text}

        Return one triple per line in this format:
        (subject, predicate, object)
        """)

        chain = LLMChain(
            llm=ChatOpenAI(temperature=0),
            prompt=prompt
        )
        result = chain.run(text=text)

        triples = []
        for line in result.split("\n"):
            if not line.strip(): continue
            try:
                parts = line.strip("()").split(",")
                subject, predicate, obj = map(str.strip, parts)
                triples.append({
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj
                })
            except Exception as e:
                print(f"[KG] Skipped malformed triple: {line}")
        return triples

    def add_triples_to_graph(self, triples: List[Dict[str, str]]):
        with self._driver.session(database=self.database) as session:
            for triple in triples:
                # Generate unique IDs based on the entity names
                subject_id = hashlib.sha256(triple['subject'].encode()).hexdigest()[:12]
                object_id = hashlib.sha256(triple['object'].encode()).hexdigest()[:12]
                
                # Create or update entities with IDs and infer types
                session.run(
                    """
                    MERGE (a:Entity {name: $subject})
                    SET a.id = $subject_id
                    SET a.type = CASE 
                        WHEN $subject CONTAINS 'Date' THEN 'Date'
                        WHEN $subject CONTAINS 'Price' THEN 'Price'
                        WHEN $subject CONTAINS 'Form' THEN 'Document'
                        WHEN $subject CONTAINS 'Number' THEN 'Number'
                        ELSE 'Entity'
                    END
                    MERGE (b:Entity {name: $object})
                    SET b.id = $object_id
                    SET b.type = CASE 
                        WHEN $object CONTAINS 'Date' THEN 'Date'
                        WHEN $object CONTAINS 'Price' THEN 'Price'
                        WHEN $object CONTAINS 'Form' THEN 'Document'
                        WHEN $object CONTAINS 'Number' THEN 'Number'
                        ELSE 'Entity'
                    END
                    MERGE (a)-[r:RELATES {type: $predicate}]->(b)
                    """,
                    subject=triple['subject'],
                    subject_id=subject_id,
                    object=triple['object'],
                    object_id=object_id,
                    predicate=triple['predicate']
                )

    async def update_graph(self, text_chunks: List[str]) -> Dict:
        """Update knowledge graph with new information from text chunks using LangChain."""
        if not self._driver:
            raise Exception("[NEO4J] No database connection available")
        
        try:
            print(colored("\n[KG] Starting knowledge graph update...", "magenta"))

            # Find the highest existing chunk_id
            with self._driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (c:TextChunk)
                    RETURN COALESCE(MAX(c.chunk_id), -1) as max_id
                """)
                start_id = result.single()["max_id"] + 1
                print(colored(f"\n[KG] Starting chunk IDs from {start_id}", "magenta"))

            # Convert text chunks to LangChain documents with unique IDs
            documents = [
                Document(page_content=chunk, metadata={"chunk_id": i + start_id})
                for i, chunk in enumerate(text_chunks)
            ]

            print(colored(f"\n[KG] Processing {len(documents)} documents:", "magenta"))
            for i, doc in enumerate(documents):
                print(colored(f"\n[KG] Document {i+1}:", "magenta"))
                print(colored(f"Chunk ID: {doc.metadata['chunk_id']}", "magenta"))
                print(colored(f"Content:\n{doc.page_content}", "magenta"))

            # Store text chunks in Neo4j vector index
            print(colored("\n[KG] Adding documents to Neo4j vector store...", "magenta"))
            self.vector_store = Neo4jVector.from_documents(
                documents,
                self.embeddings,
                url=self.uri,
                username=self.username,
                password=self.password,
                database=self.database,
                index_name="text_chunks",
                node_label="TextChunk",
                text_node_property="content",
                embedding_node_property="embedding",
                search_type="hybrid"
            )

            print(colored(f"\n[KG] Successfully added {len(documents)} documents to Neo4j vector store", "magenta"))

            # Extract triples and insert them as Entity-RELATES->Entity
            total_triples = 0
            for i, chunk in enumerate(text_chunks):
                print(colored(f"\n[KG] Extracting triples from chunk {i+1}:", "magenta"))
                print(colored(f"Content:\n{chunk}", "magenta"))
                triples = self.extract_triples_from_text(chunk)
                print(colored(f"Extracted {len(triples)} triples:", "magenta"))
                for triple in triples:
                    print(colored(f"  • Subject: {triple['subject']}", "magenta"))
                    print(colored(f"    Predicate: {triple['predicate']}", "magenta"))
                    print(colored(f"    Object: {triple['object']}", "magenta"))
                self.add_triples_to_graph(triples)
                total_triples += len(triples)

            print(colored(f"\n[KG] Added total of {total_triples} relationships to Neo4j", "magenta"))

            # Get updated graph stats
            stats = self._get_graph_stats()
            print(colored("\n[KG] Final knowledge graph statistics:", "magenta"))
            print(colored(f"  • Total nodes: {stats['num_nodes']}", "magenta"))
            print(colored(f"  • Total edges: {stats['num_edges']}", "magenta"))
            print(colored("  • Node types:", "magenta"))
            for type_, count in stats.get('node_types', {}).items():
                print(colored(f"    - {type_}: {count}", "magenta"))
            print(colored("  • Relationship types:", "magenta"))
            for type_, count in stats.get('relation_types', {}).items():
                print(colored(f"    - {type_}: {count}", "magenta"))
            return stats

        except Exception as e:
            print(colored(f"\n[KG] Error updating knowledge graph: {str(e)}", "red"))
            raise

    def _get_graph_stats(self) -> Dict:
        """Get graph statistics from Neo4j."""
        if not self._driver:
            raise Exception("[NEO4J] No database connection available")
            
        with self._driver.session(database=self.database) as session:
            # Execute queries
            stats = {}
            queries = {
                "num_nodes": "MATCH (n:Entity) RETURN count(n) as count",
                "num_edges": "MATCH ()-[r:RELATES]->() RETURN count(r) as count",
                "node_types": """
                    MATCH (n:Entity)
                    WITH n.type AS type, count(*) AS count
                    WHERE type IS NOT NULL
                    RETURN type, count
                """,
                "relation_types": """
                    MATCH ()-[r:RELATES]->()
                    WITH r.type AS type, count(*) AS count
                    WHERE type IS NOT NULL
                    RETURN type, count
                """
            }
            
            for key, query in queries.items():
                result = session.run(query)
                if key in ["num_nodes", "num_edges"]:
                    stats[key] = result.single()["count"]
                else:
                    stats[key] = {record["type"]: record["count"] for record in result}
            
        return stats

    async def search_graph(self, query: str) -> Dict:
        """Search the knowledge graph using natural language queries."""
        if not self._driver:
            raise Exception("[NEO4J] No database connection available")
            
        try:
            print(colored(f"\n[KG] Starting graph search for query: '{query}'", "magenta"))
            
            # Get answer from QA chain
            print(colored("\n[KG] Getting answer from QA chain...", "magenta"))
            qa_result = self.qa_chain.run(query)
            print(colored(f"\n[KG] QA Chain answer:\n{qa_result}", "magenta"))
            
            # Get similar chunks with scores using hybrid search
            results = []
            if self.vector_store:
                try:
                    print(colored("\n[KG] Performing vector similarity search...", "magenta"))
                    # Get similar documents with scores
                    similar_docs = self.vector_store.similarity_search_with_score(query, k=5)
                    
                    # Process each result
                    print(colored(f"\n[KG] Found {len(similar_docs)} similar documents:", "magenta"))
                    for i, (doc, score) in enumerate(similar_docs):
                        if isinstance(doc, Document):
                            result = {
                                "content": doc.page_content,
                                "metadata": doc.metadata,
                                "score": float(score)  # Ensure score is a float
                            }
                            results.append(result)
                            print(colored(f"\n[KG] Result {i+1}:", "magenta"))
                            print(colored(f"Score: {score:.4f}", "magenta"))
                            print(colored(f"Metadata: {doc.metadata}", "magenta"))
                            print(colored(f"Content:\n{doc.page_content}", "magenta"))
                        else:
                            print(colored(f"\n[KG] Warning: Unexpected document format: {type(doc)}", "yellow"))
                except Exception as e:
                    print(colored(f"\n[KG] Warning: Vector search failed: {str(e)}", "yellow"))
            
            # Compile results
            result = {
                "answer": qa_result,
                "results": results,
                "query": query
            }
            
            print(colored(f"\n[KG] Search complete - Found {len(results)} relevant chunks", "magenta"))
            return result
            
        except Exception as e:
            print(colored(f"\n[KG] Error searching graph: {str(e)}", "red"))
            raise

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()

    def get_graph_data(self) -> Dict:
        """Get the current state of the knowledge graph for visualization."""
        if not self._driver:
            raise Exception("[NEO4J] No database connection available")
            
        try:
            print(colored("[KG] Retrieving graph data for visualization...", "magenta"))
            
            with self._driver.session(database=self.database) as session:
                # Get all nodes with their IDs and types
                node_result = session.run("""
                    MATCH (n:Entity)
                    WHERE n.name IS NOT NULL
                    RETURN 
                        COALESCE(n.id, n.name) as id,
                        n.name as name,
                        COALESCE(n.type, 'Entity') as type
                """)
                nodes = [{"id": record["id"], "name": record["name"], "type": record["type"]} 
                        for record in node_result]

                # Get all relationships using the node IDs
                link_result = session.run("""
                    MATCH (s:Entity)-[r:RELATES]->(t:Entity)
                    WHERE s.name IS NOT NULL AND t.name IS NOT NULL
                    RETURN 
                        COALESCE(s.id, s.name) as source,
                        COALESCE(t.id, t.name) as target,
                        r.type as type
                """)
                links = [{"source": record["source"], "target": record["target"], "type": record["type"]}
                        for record in link_result]

            print(colored(f"[KG] Retrieved {len(nodes)} nodes and {len(links)} relationships", "magenta"))
            return {
                "nodes": nodes,
                "links": links
            }
        except Exception as e:
            print(colored(f"[KG] Error retrieving graph data: {str(e)}", "red"))
            raise

    def clear(self) -> None:
        """Clear all data from the knowledge graph and vector store."""
        if not self._driver:
            raise Exception("[NEO4J] No database connection available")
            
        try:
            print(colored("[KG] Clearing all data from knowledge graph...", "magenta"))
            
            with self._driver.session(database=self.database) as session:
                # Delete all relationships and nodes
                session.run("MATCH (n) DETACH DELETE n")
                
                try:
                    # Try to drop the vector index using the newer procedure name
                    session.run("""
                        CALL vector.index.drop('text_chunks')
                    """)
                except Exception as e:
                    # If the index doesn't exist or can't be dropped, just log it
                    print(colored("[KG] Note: Vector index could not be dropped (may not exist)", "yellow"))
            
            # Reset vector store reference
            self.vector_store = None
            
            print(colored("[KG] Knowledge graph cleared successfully", "magenta"))
        except Exception as e:
            print(colored(f"[KG] Error clearing knowledge graph: {str(e)}", "red"))
            raise 