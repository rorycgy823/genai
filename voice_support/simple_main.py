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

# Read PDF
reader = PdfReader(r"G:\Data_Science\github_project\voice_support\kfs_mtg.pdf")
text = "".join(page.extract_text() for page in reader.pages)

# Initialize embedding model
embedding_function = SentenceTransformerEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
)

chunks = text_splitter.split_text(text)
documents = [Document(page_content=chunk, metadata={"source": "pdf"}) for chunk in chunks]

# Create vector store
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_function,
    persist_directory="./chroma_db"
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create custom prompt template
custom_prompt = PromptTemplate(
    template="""基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明无法从提供的文档中找到答案。

上下文: {context}

问题: {question}

回答:""",
    input_variables=["context", "question"]
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=Tongyi(api_key="sk-015ea57c8b254c4181d30b2de4259d8b", model="qwen-max"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": custom_prompt
    }
)

print("Simple QA system initialized successfully!")
