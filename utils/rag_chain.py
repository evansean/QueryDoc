from langchain_ollama.llms import OllamaLLM
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.conversation.base import ConversationChain 
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.memory import ConversationSummaryMemory
from langchain_classic.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
from huggingface_hub import login

load_dotenv()
login(token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])

def create_memory(llm):
    """Initialize conversation summary memory."""
    return ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        input_key="question",
        return_messages=False
    )

def summarize_docs(docs, llm):
    """Summarize the content of documents using RAG."""
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)

    # Run the summarization
    summary = chain.invoke(docs)
    return summary['output_text']

def run_rag_query(vectordb, query, memory, llm):
    """Run a RAG query against the vector store."""

    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})

    docs = retriever.invoke(query)

    # Format context with source info
    context_parts = []
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[Source: {src}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful and friendly assistant. "
            "Answer user questions based on the documents and past conversation. "
            "If the documents don't contain the answer, use your own knowledge. "
            "Keep responses natural and concise, as if talking to a human."
        ),
        HumanMessagePromptTemplate.from_template(
            "Previous conversation:\n{chat_history}\n\n"
            "Document excerpts (with sources):\n{context}\n\n"
            "User question:\n{question}\n\n"
            "Answer naturally and directly. "
            "Cite sources when relevant."
        ),
    ])

    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
        memory=memory,
        verbose=True,
    )

    response = chain.invoke({"question": query, "context": context})
    return response['text']