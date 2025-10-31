# 🧠 QueryDoc — Multi-Source Conversational RAG System

<img width="1919" height="982" alt="image" src="https://github.com/user-attachments/assets/26afb330-45b9-457c-855c-ccf239967cc0" />

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://evansean-querydoc.streamlit.app/)

QueryDoc is an interactive document assistant that lets you **chat, summarize, and retrieve insights from multiple documents and websites** — all in one conversational interface.

It demonstrates a **full Retrieval-Augmented Generation (RAG) workflow**:
📥 multi-format ingestion → 🧱 embedding & metadata tagging → 🔍 adaptive vector retrieval → ⚖️ cross-encoder reranking → 💬 memory-aware conversational reasoning.

---

## 🚀 Features

- **📂 Multi-format ingestion** — Upload and process PDF, TXT, DOCX, CSV, and even URLs. All sources are parsed and prepared for downstream retrieval.  
- **⚙️ Automatic parsing & embedding** — Documents are chunked, embedded using HuggingFace / DeepSeek-V3, and stored in Chroma with descriptive metadata.  
- **🏷️ Metadata-driven categorization** — Every chunk includes metadata such as file name, type, and source URL, enabling structured organization.  
- **🎯 Metadata filtering & multi-source querying** — Users can select specific documents or URLs; the retriever dynamically filters results based on these selections.  
- **🔍 Two-stage retrieval** — 
  1. **Adaptive vector retrieval (MMR)** selects top relevant chunks using `top_k` and `fetch_k`, which adapt to the number of document chunks for better coverage and efficiency.
  2.  **Cross-encoder reranking** — After the initial MMR retrieval, the top candidate chunks are passed to a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which scores each query-document pair directly.  
         - This stage evaluates **semantic relevance at a fine-grained level**, considering the full interaction between the query and each document chunk.  
         - By reranking the retrieved chunks, it ensures that the LLM receives the **most contextually relevant and diverse information**, improving answer accuracy.  
- **📚 Per-document summarization** — Each document is summarized individually using a map-reduce summarization chain, providing concise overviews before multi-source fusion.  
- **💬 Multi-source conversational chat** — Ask questions across selected sources; retrieved context from different documents is fused to generate unified answers.  
- **🧠 Memory-aware conversations** — Maintains multi-turn chat history using `ConversationSummaryMemory`, ensuring context retention across user queries.  




[🚀 Try QueryDoc Live on Streamlit](https://evansean-querydoc.streamlit.app/)

---

## 💡 Example Usage Flow

1. Upload files or enter URLs (PDF, Word, CSV, TXT, websites).  
2. Documents are parsed, chunked, embedded, and stored in Chroma with metadata.  
3. Auto-generate **per-document summaries**.  
4. Select which sources to chat with (multi-document queries supported).  
5. Ask questions naturally — the system retrieves relevant chunks using **adaptive MMR**, reranks top chunks with **CrossEncoder**, and generates context-aware answers.  

---

## 🧾 Setup & Run

```bash
# Clone the repository
git clone https://github.com/evansean/QueryDoc.git
cd QueryDoc

# Install dependencies
pip install -r requirements.txt

# Set your Hugging Face token in Streamlit secrets or .env
HUGGINGFACEHUB_API_TOKEN=<your_token>

# Run the app
streamlit run app.py

