# 🎥 RAGtube — Chat with YouTube Videos

<img width="1919" height="975" alt="image" src="https://github.com/user-attachments/assets/12261a86-5312-4d70-8366-aaec5800eb40" />

**RAGtube** is an interactive video assistant that allows users to **ask questions about YouTube videos** and receive accurate, conversational answers. Unlike traditional video summaries or static transcripts, RAGtube combines **retrieval-augmented generation (RAG)** with **multi-turn conversational memory** to provide context-aware responses. It can answer questions based on the video content, recall details from prior conversation turns, and even respond to general knowledge questions that go beyond the video.


## 🧭 How to Use

1. Paste a YouTube video URL into the input box.  
2. Click **“Load Video”** to load the video’s transcript.  
3. Ask questions naturally in the chat box — e.g.:
   - “What is this video about?”
   - “What are some highlights in this video?”
   - “Who is the speaker talking about?”
4. The assistant will retrieve relevant transcript parts and answer conversationally.

## Core Features

1. **YouTube Video Transcript Processing**
   - Fetches and parses YouTube video transcripts.
   - Splits transcripts into manageable chunks suitable for vector embedding.

2. **Vector Store & Retrieval**
   - Implements a **Chroma vector database** to store document embeddings.
   - Uses **MMR (Maximal Marginal Relevance)** for retrieval:
     - MMR ensures that the retrieved documents are both relevant and diverse, reducing redundancy in answers.
     - The retriever fetches the most pertinent chunks of the transcript in response to a query.

3. **RAG — Retrieval-Augmented Generation**
   - Uses an **LLM (Ollama’s LLaMA 3 or Deepseek V3.1)** to generate answers based on retrieved documents.
   - Combines **video context** with **conversation history** for coherent, multi-turn dialogues.

4. **Conversational Memory**
   - Maintains **conversation summaries** so that the assistant can remember details from previous questions and answers.
   - Supports multi-turn interaction without losing context.

5. **General Knowledge Integration**
   - The assistant can also answer questions beyond the video content, leveraging the LLM’s inherent knowledge.

6. **Streamlit Frontend**
   - Provides an **interactive chat interface** alongside the video and its summary.
   - Chat window is scrollable, while the video and summary remain static.
   - Users can ask questions in a natural, conversational manner.


## 💻 Run Locally

Follow these steps to set up and run **RAGtube** with **Deepseek** on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/evansean/ragtube.git
cd ragtube
```

### 2. Create and Activate Conda Environment
```bash
conda create -n {environment_name} python=3.10
conda activate {environment_name}
```

### 3. Install Dependencies in Conda Environment
```bash
pip install -r requirements.txt
```

### 4. Set up Environment Variables
- Create a .env file in the project root directory and add the following:
```bash
HUGGINGFACEHUB_API_TOKEN = {YOUR_HF_API_TOKEN}
```

### 5. Run the app
```bash
streamlit run app.py
```

