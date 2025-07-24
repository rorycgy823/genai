import os
import re
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Tongyi
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import json

# Read PDF
# TODO: Update this path to your PDF file
pdf_path = "kfs_mtg.pdf"  # Place your PDF file in the same directory
reader = PdfReader(pdf_path)
text = "".join(page.extract_text() for page in reader.pages)

# Initialize embedding model
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embedding_function = SentenceTransformerEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')

class GraphRAG:
    def __init__(self, documents: List[str], embedding_model, chunk_size: int = 500, chunk_overlap: int = 50):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.graph = nx.Graph()
        self.chunks = []
        self.chunk_embeddings = []
        self.entities = {}
        self.relationships = []
        
        # Process documents
        self._process_documents(documents)
        self._build_knowledge_graph()
        
    def _process_documents(self, documents: List[str]):
        """Split documents into chunks and create embeddings"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        
        for doc in documents:
            chunks = text_splitter.split_text(doc)
            self.chunks.extend(chunks)
        
        # Create embeddings for all chunks
        self.chunk_embeddings = self.embedding_model.encode(self.chunks)
        
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text using simple pattern matching"""
        # Extract potential entities (words with specific patterns)
        entities = []
        
        # Extract Chinese names/organizations (2-4 characters)
        chinese_entities = re.findall(r'[\u4e00-\u9fff]{2,4}(?:公司|银行|保险|基金|证券|投资|集团|有限|股份)', text)
        entities.extend(chinese_entities)
        
        # Extract numbers with units (amounts, percentages, etc.)
        number_entities = re.findall(r'\d+(?:\.\d+)?(?:%|元|万|亿|千|百)', text)
        entities.extend(number_entities)
        
        # Extract dates
        date_entities = re.findall(r'\d{4}年\d{1,2}月\d{1,2}日|\d{4}年\d{1,2}月|\d{4}年', text)
        entities.extend(date_entities)
        
        # Extract key financial terms
        financial_terms = re.findall(r'(?:资产|负债|收入|支出|利润|亏损|投资|融资|贷款|存款|利率|汇率)', text)
        entities.extend(financial_terms)
        
        return list(set(entities))
    
    def _build_knowledge_graph(self):
        """Build knowledge graph from chunks"""
        print("Building knowledge graph...")
        
        # Extract entities from each chunk
        for i, chunk in enumerate(self.chunks):
            entities = self._extract_entities(chunk)
            
            # Add entities to graph
            for entity in entities:
                if entity not in self.entities:
                    self.entities[entity] = []
                self.entities[entity].append(i)
                self.graph.add_node(entity, type='entity')
            
            # Add chunk as node
            chunk_id = f"chunk_{i}"
            self.graph.add_node(chunk_id, type='chunk', content=chunk)
            
            # Connect entities to chunks
            for entity in entities:
                self.graph.add_edge(entity, chunk_id, relation='contains')
        
        # Create relationships between entities based on co-occurrence
        for i, chunk in enumerate(self.chunks):
            entities = self._extract_entities(chunk)
            for j, entity1 in enumerate(entities):
                for entity2 in entities[j+1:]:
                    if self.graph.has_edge(entity1, entity2):
                        self.graph[entity1][entity2]['weight'] += 1
                    else:
                        self.graph.add_edge(entity1, entity2, relation='co_occurs', weight=1)
        
        print(f"Knowledge graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def _get_relevant_entities(self, query: str, top_k: int = 5) -> List[str]:
        """Get entities most relevant to the query"""
        query_embedding = self.embedding_model.encode([query])
        
        relevant_entities = []
        for entity in self.entities.keys():
            entity_embedding = self.embedding_model.encode([entity])
            similarity = cosine_similarity(query_embedding, entity_embedding)[0][0]
            relevant_entities.append((entity, similarity))
        
        # Sort by similarity and return top_k
        relevant_entities.sort(key=lambda x: x[1], reverse=True)
        return [entity for entity, _ in relevant_entities[:top_k]]
    
    def _graph_walk(self, start_entities: List[str], max_depth: int = 2) -> List[str]:
        """Perform graph walk to find related chunks"""
        visited_chunks = set()
        
        for entity in start_entities:
            if entity in self.graph:
                # BFS to find connected chunks
                queue = [(entity, 0)]
                visited = set([entity])
                
                while queue:
                    current_node, depth = queue.pop(0)
                    
                    if depth >= max_depth:
                        continue
                    
                    for neighbor in self.graph.neighbors(current_node):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            
                            # If neighbor is a chunk, add it to results
                            if neighbor.startswith('chunk_'):
                                visited_chunks.add(neighbor)
                            else:
                                queue.append((neighbor, depth + 1))
        
        return list(visited_chunks)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant chunks using GraphRAG"""
        # Step 1: Get relevant entities
        relevant_entities = self._get_relevant_entities(query, top_k=10)
        
        # Step 2: Perform graph walk to find connected chunks
        relevant_chunk_ids = self._graph_walk(relevant_entities, max_depth=2)
        
        # Step 3: Get chunk indices and compute similarity with query
        query_embedding = self.embedding_model.encode([query])
        chunk_similarities = []
        
        for chunk_id in relevant_chunk_ids:
            chunk_idx = int(chunk_id.split('_')[1])
            chunk_embedding = self.chunk_embeddings[chunk_idx].reshape(1, -1)
            similarity = cosine_similarity(query_embedding, chunk_embedding)[0][0]
            chunk_similarities.append((chunk_idx, similarity))
        
        # Step 4: If no chunks found through graph walk, fallback to similarity search
        if not chunk_similarities:
            # Fallback to traditional similarity search
            similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            chunk_similarities = [(idx, similarities[idx]) for idx in top_indices]
        
        # Step 5: Sort by similarity and return top chunks
        chunk_similarities.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [self.chunks[idx] for idx, _ in chunk_similarities[:top_k]]
        
        return top_chunks

# Simple retriever class that doesn't inherit from BaseRetriever
class SimpleGraphRAGRetriever:
    def __init__(self, graph_rag: GraphRAG):
        self.graph_rag = graph_rag
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for the query"""
        chunks = self.graph_rag.retrieve(query, top_k=5)
        documents = [Document(page_content=chunk, metadata={"source": "pdf"}) for chunk in chunks]
        return documents

# Initialize GraphRAG system
print("Initializing GraphRAG system...")
graph_rag = GraphRAG([text], embedding_model, chunk_size=500, chunk_overlap=50)

# Create simple retriever
graph_retriever = SimpleGraphRAGRetriever(graph_rag)

# Create custom prompt template
custom_prompt = PromptTemplate(
    template="""基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明无法从提供的文档中找到答案。

上下文: {context}

问题: {question}

回答:""",
    input_variables=["context", "question"]
)

# Create QA chain with GraphRAG - using a simple approach
from langchain.chains.question_answering import load_qa_chain

# Create the LLM
# TODO: Replace with your actual Tongyi API key
llm = Tongyi(api_key="YOUR_TONGYI_API_KEY_HERE", model="qwen-max")

# Create QA chain
qa_chain_base = load_qa_chain(llm, chain_type="stuff", prompt=custom_prompt)

# Create a wrapper function for the QA chain
def qa_chain_function(query_dict):
    """Wrapper function for QA chain"""
    try:
        query = query_dict.get("query", "")
        
        # Get relevant documents using GraphRAG
        documents = graph_retriever.get_relevant_documents(query)
        
        # Run the QA chain
        result = qa_chain_base({"input_documents": documents, "question": query})
        
        # Format the response
        answer = result.get("output_text", "抱歉，无法找到相关答案。")
        
        # Add source information
        if documents:
            answer += f"\n\n参考来源: 基于{len(documents)}个相关文档片段"
        
        return {"result": answer, "source_documents": documents}
    except Exception as e:
        return {"result": f"处理问题时出现错误: {str(e)}", "source_documents": []}

# Create a simple class to mimic RetrievalQA behavior
class SimpleQAChain:
    def __init__(self, qa_function):
        self.qa_function = qa_function
    
    def __call__(self, query_dict):
        return self.qa_function(query_dict)
    
    def run(self, query):
        if isinstance(query, str):
            result = self.qa_function({"query": query})
        else:
            result = self.qa_function(query)
        return result.get("result", "抱歉，无法找到相关答案。")

# Create the final QA chain
qa_chain = SimpleQAChain(qa_chain_function)

print("GraphRAG system initialized successfully!")
