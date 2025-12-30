import streamlit as st
import tempfile
import re
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Offline RAG 2025", layout="wide")
st.title("🧠 Offline RAG + Context Retrieval")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("Indexing PDF..."):
        # Use LangChain's native loader for better metadata handling
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        os.remove(tmp_path) # Clean up

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(pages)

        # 2025 Best Practice: Use langchain-huggingface partner package
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
    st.success("PDF indexed successfully")

question = st.text_input("Ask a question")

if question and st.session_state.vectorstore:
    with st.spinner("Retrieving context..."):
        docs_and_scores = st.session_state.vectorstore.similarity_search_with_score(question, k=3)
        docs, scores = zip(*docs_and_scores)
        context = "\n\n".join([doc.page_content for doc in docs])
        avg_score = sum(scores) / len(scores)

    st.subheader("📄 Retrieved Context")
    st.write(context)
    
    st.subheader("📊 Relevance Score")
    relevance = "🟢 HIGH" if avg_score < 0.4 else "🟠 MEDIUM" if avg_score < 0.6 else "🔴 LOW"
    st.write(f"{relevance} relevance (score: {avg_score:.2f})")
    
    st.subheader("💡 Suggestion")
    st.write("Use the context above to answer your question. For best results, verify the information against your document.")
