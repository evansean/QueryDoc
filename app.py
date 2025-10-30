import streamlit as st
from streamlit_chat import message
from utils.loader import load_docs
from utils.vector_store import create_vectorstore
from utils.rag_chain import summarize_docs, run_rag_query, create_memory
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from huggingface_hub import login, HfApi
from dotenv import load_dotenv
import os
from collections import defaultdict

# --- SETUP ---
load_dotenv()
login(token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])
api = HfApi(token=os.getenv("HF_TOKEN"))

# --- PAGE CONFIG ---
st.set_page_config(page_title="📄 QueryDoc — Chat with Your Documents", layout="wide")

# --- SESSION STATE ---
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "llm" not in st.session_state:
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V3.1",
        task="conversational",
        temperature=0,
        provider="auto",
    )
    st.session_state.llm = ChatHuggingFace(llm=llm)
if "memory" not in st.session_state:
    st.session_state.memory = create_memory(st.session_state.llm)

# --- SIDEBAR ---
st.sidebar.header("📂 Upload Documents or Add URLs")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more files",
    type=["pdf", "docx", "doc", "txt", "csv"],
    accept_multiple_files=True,
)

url_input = st.sidebar.text_area("Enter one or more URLs (one per line):")
urls = [u.strip() for u in url_input.splitlines() if u.strip()] if url_input else None

load_btn = st.sidebar.button("Load Documents")



if load_btn and (uploaded_files or urls):
    st.write("Uploaded files:", uploaded_files)
    st.write("URLs:", urls)
    with st.spinner("📥 Loading documents..."):
        docs = load_docs(uploaded_files, urls)
        # Group chunks by source
        docs_by_file = defaultdict(list)
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            docs_by_file[source].append(doc)


    with st.spinner("🔍 Creating vector store..."):
        vectordb = create_vectorstore(docs)

    with st.spinner("🧠 Summarizing documents..."):
        # summary = summarize_docs(docs, st.session_state.llm)
        summaries = summarize_docs(docs, st.session_state.llm)

    st.session_state.vectordb = vectordb
    st.session_state.summary = summaries
    st.session_state.chat_history = []
    st.session_state.memory.clear()
    st.sidebar.success("✅ Documents loaded successfully!")

# --- MAIN LAYOUT ---
st.title("📄 QueryDoc — Chat with Your Documents")

if st.session_state.vectordb:
    col1, col2 = st.columns([2, 2])

    with col1:
        st.markdown("### 📝 Document Summaries")

        selected_file = st.selectbox("Select a document to view its summary:", list(st.session_state.summary.keys()))

        st.write(st.session_state.summary[selected_file])

    with col2:
        all_sources = list(st.session_state.summary.keys())  # from your summarize_docs function
        selected_sources = st.multiselect("Select sources to include in chat", options=all_sources, default=all_sources)

        chat_container = st.container(height=500)
        with chat_container:
            for i, turn in enumerate(st.session_state.chat_history):
                message(turn["user"], is_user=True, key=f"user_{i}")
                message(turn["assistant"], key=f"assistant_{i}")

        def handle_user_query():
            query = st.session_state.user_query.strip()
            if query and st.session_state.vectordb:
                with st.spinner("Thinking..."):
                    response = run_rag_query(
                        st.session_state.vectordb,
                        query,
                        st.session_state.memory,
                        st.session_state.llm,
                        sources=selected_sources
                    )
                st.session_state.chat_history.append({"user": query, "assistant": response})
                st.session_state.user_query = ""

        st.text_input(
            "Ask a question:",
            key="user_query",
            on_change=handle_user_query,
            placeholder="Type your question and press Enter...",
        )

else:
    st.info("👈 Upload documents or paste URLs in the sidebar to begin.")
