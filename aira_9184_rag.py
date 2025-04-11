
# meet AiRA-9184 (Artificial Intelligence for RA-9184)

import os
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. Load and split PDF
pdf_path = r"D:\\LEARNING MANAGEMENT\\AiRA-9184\\AiRA-9184\\data\\Updated-2016-Revised-IRR-of-RA-No.-9184-as-of-19-July-2024.pdf"
print("Loading PDF...")
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"Loaded {len(docs)} pages.")

# 2. Split the document into manageable chunks
print("Splitting document into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)
print(f"Split into {len(split_docs)} chunks.")

# 3. Use a fast embedding model for vector store (must match FAISS index)
print("Initializing embeddings with a faster model...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Load or build the FAISS index
faiss_path = "faiss_index"
if os.path.exists(faiss_path):
    print("Loading cached FAISS index...")
    vectorstore = FAISS.load_local(
        faiss_path, embeddings, allow_dangerous_deserialization=True)
else:
    print("Creating FAISS index (first time)...")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(faiss_path)
    print("FAISS index saved for future use.")

# 5. Setup retriever from vector store
print("Setting up retriever...")
retriever = vectorstore.as_retriever()
print("Retriever ready.")

# 6. Use DeepSeek for answering questions (LLM only, not for embeddings)
print("Setting up LLM and RetrievalQA chain...")
llm = OllamaLLM(model="deepseek-r1")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
print("LLM and RetrievalQA chain ready.")

# 7. Interactive Q&A loop
while True:
    query = input(
        "Ask something about AiRA-9184 (Artificial Intelligence for RA-9184) (or type 'exit'): ")
    if query.lower() == 'exit':
        break
    print(f"Processing query: {query}")
    answer = qa_chain.invoke(query)
    print("Answer:", answer)
