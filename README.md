# 🧠 QueryDoc — Multi-Source Conversational RAG System

QueryDoc is an interactive document assistant that lets you chat, summarize, and retrieve insights from multiple documents and websites — all in one conversational interface.

It demonstrates a full Retrieval-Augmented Generation (RAG) workflow:  
📥 multi-format ingestion → 🧱 embedding & metadata tagging → 🔍 selective retrieval → 💬 conversational reasoning — powered by DeepSeek-V3, LangChain, and Chroma.

---

## 🚀 Features
- 📂 **Multi-format ingestion** — Upload and process PDF, TXT, DOCX, CSV, and even URLs.  
- ⚙️ **Automatic parsing & embedding** — Each source is parsed, chunked, embedded, and stored in Chroma with descriptive metadata.  
- 🏷️ **Metadata-driven categorization** — Every document chunk includes metadata like file name, type, and source URL.  
- 🎯 **Metadata filtering** — Users can select specific documents or URLs to query; the retriever dynamically filters results based on those metadata fields.  
- 📚 **Per-document summarization** — Each document is summarized individually using a map-reduce summarization chain for concise overviews.  
- 💬 **Multi-source conversational chat** — Ask questions across multiple selected sources or all of them at once. The model fuses retrieved context from different documents to generate unified answers.  
- 🧠 **Memory-aware conversations** — Maintains multi-turn chat history with ConversationSummaryMemory, allowing context retention across user questions.  
- 🔍 **Contextual retrieval (MMR)** — Uses Maximal Marginal Relevance to balance diversity and similarity for retrieved chunks.  
- 🧩 **Modular and extensible** — Clean function-based architecture for ingestion, summarization, and retrieval.  
- (Coming soon) ⚖️ **Cross-encoder reranking** — for even finer-grained result reordering after vector retrieval.

---

## 🧩 Architecture Overview
      ┌──────────────────────────────┐
      │     Uploaded Files & URLs    │
      └──────────────┬───────────────┘
                     │
            Data Parsing Layer
  (PDF, DOCX, TXT, CSV, Webpage Extraction)
                     │
                     ▼
         Text Chunking & Embedding
      (LangChain + DeepSeek Embeddings)
                     │
                     ▼
     🧱 Chroma Vector Database with Metadata
     (Stores embeddings + {source, type, name})
                     │
                     ▼
        Retriever (MMR + Metadata Filters)
     → user-selected sources (multi-doc queries)
                     │
                     ▼
  RAG Chain + ConversationSummaryMemory
    (DeepSeek-V3 via Hugging Face Endpoint)
                     │
                     ▼
            Streamlit Interface
   (Summaries, Source Selection, and Chat)

---

## 🧩 How Each Component Meets the RAG Engineering Goals

| Requirement | Implementation in QueryDoc |
|-------------|----------------------------|
| **Data parsing from various formats** | Supports TXT, PDF, DOCX, CSV, and website URLs via Unstructured and LangChain loaders. |
| **Text embedding and vector DB integration** | Uses LangChain embeddings stored in Chroma, ensuring semantic search and fast retrieval. |
| **Categorization within vector DB** | Embeddings are tagged with metadata (source type, name, URL) for structured organization. |
| **Metadata filtering and multi-source querying** | The retriever applies metadata filters to limit results to selected files/URLs — enabling dynamic source selection at runtime. |
| **Efficient retrieval with memory management** | Combines MMR retriever with ConversationSummaryMemory to maintain both relevance and conversational flow. |
| **End-to-end data injection, parsing, and response generation** | From parsing → embedding → metadata tagging → retrieval → LLM answer generation, all in one seamless pipeline. |

---

## 💡 Example Usage Flow
1. Upload files or enter URLs — load PDFs, Word docs, CSVs, or websites.  
2. The app parses and embeds each source into the Chroma vector database with metadata.  
3. Auto-generate summaries for each uploaded document.  
4. Select which sources to chat with using checkboxes or dropdowns.  
5. Ask questions naturally — the system retrieves relevant chunks only from selected sources and includes past chat context in responses.

---

## 🧰 Tech Stack

| Layer | Tools / Libraries |
|-------|------------------|
| Frontend | 🖥️ Streamlit |
| LLM Backend | 🤖 DeepSeek-V3 via Hugging Face API |
| Framework | 🦜 LangChain |
| Vector Store | 🧱 Chroma |
| Memory | 🧠 ConversationSummaryMemory |
| Environment | Python 3.10, dotenv, HuggingFace Hub |

---

## 🧾 Setup & Run

```bash
# Clone the repository
git clone https://github.com/<your-username>/QueryDoc.git
cd QueryDoc

# Install dependencies
pip install -r requirements.txt

# Set your Hugging Face token in Streamlit secrets or .env
HUGGINGFACEHUB_API_TOKEN=<your_token>

# Run the app
streamlit run app.py