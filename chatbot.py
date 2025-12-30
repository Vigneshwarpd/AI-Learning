import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Load environment variables
load_dotenv()

st.set_page_config(page_title="6th Std English RAG Chatbot", layout="wide")
st.title("📘 6th Standard English Book Chatbot")

st.write("Upload your **6th Standard English textbook PDF** and ask questions.")

# Upload PDF
uploaded_file = st.file_uploader("Upload English Textbook PDF", type=["pdf"])

@st.cache_resource
def create_vector_db(pdf_path):
    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Text Splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # 3. Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # 4. Vector Store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

if uploaded_file:
    with open("book.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing book... Please wait"):
        vector_db = create_vector_db("book.pdf")

    st.success("Book processed successfully ✅")

    # 5. Retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    # 6. LLM
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
        model="gpt-3.5-turbo"
    )

    # 7. RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Chat UI
    st.subheader("💬 Ask Questions from the Book")

    user_question = st.text_input("Ask a question:")

    if user_question:
        with st.spinner("Thinking..."):
            response = qa_chain(user_question)

        st.markdown("### ✅ Answer:")
        st.write(response["result"])

        with st.expander("📄 Source from Book"):
            for doc in response["source_documents"]:
                st.write(doc.page_content)