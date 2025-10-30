import streamlit as st
from streamlit_chat import message
from utils.loader import load_youtube_transcript
from utils.vector_store import create_vectorstore
from utils.rag_chain import summarize_video, run_rag_query
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
from utils.rag_chain import create_memory
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os
from huggingface_hub import login, HfApi
import os

login(token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])
api = HfApi(token=os.getenv("HF_TOKEN"))
load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAGtube 🎥", layout="wide")

# --- INITIAL SESSION STATE ---
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "video_url" not in st.session_state:
    st.session_state.video_url = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if 'conversation' not in st.session_state:
    st.session_state['conversation'] =None
if "llm" not in st.session_state:
    # st.session_state.llm = ChatOllama(model="llama3", temperature=0)
    llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1",  # Chat model
    task="conversational",
    temperature= 0,
    provider="auto",
    # huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
)
    st.session_state.llm = ChatHuggingFace(llm=llm)
if "memory" not in st.session_state:
    st.session_state.memory = create_memory(st.session_state.llm)



# --- SIDEBAR ---
st.sidebar.header("🎬 RAGtube Settings")
url_input = st.sidebar.text_input("YouTube Video URL", placeholder="Paste a YouTube link...")
load_btn = st.sidebar.button("Load Video")

if load_btn and url_input:
    with st.spinner("Fetching transcript..."):
        docs = load_youtube_transcript(url_input)
    st.success("Transcript fetched.")

    with st.spinner("Building vector store..."):
        vectordb = create_vectorstore(docs)
    st.success("Vector store built.")

    with st.spinner("Summarizing video..."):
        summary = summarize_video(docs, st.session_state.llm)
    st.success("Video summarized.")

    st.session_state.vectordb = vectordb
    st.session_state.video_url = url_input
    st.session_state.summary = summary
    st.session_state.chat_history = []
    st.session_state.memory.clear()
    st.sidebar.success("Video loaded successfully!")

# --- MAIN LAYOUT ---
st.title("🎥 RAGtube — Chat with YouTube Videos")

if st.session_state.video_url:
    col1, col2 = st.columns([2, 2])

    with col1:
        st.video(st.session_state.video_url)
        st.markdown("### 📝 Video Summary")
        st.write(st.session_state.summary)

    with col2:
        # Scrollable chat area
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
                        st.session_state.llm
                    )
                st.session_state.chat_history.append({"user": query, "assistant": response})
                st.session_state.user_query = ""  

        st.text_input(
            "Ask a question:",
            key="user_query",
            on_change=handle_user_query,  # Runs before rerun
            placeholder="Type your question and press Enter...",
        )

else:
    st.info("👈 Enter a YouTube URL in the sidebar to begin.")
