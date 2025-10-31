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
from sentence_transformers import CrossEncoder

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

# for amalgamated summarization
# def summarize_docs(docs, llm):
#     """Summarize the content of documents using RAG."""
#     chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)

#     # Run the summarization
#     summary = chain.invoke(docs)
#     return summary['output_text']

def summarize_docs(docs, llm):
    """
    Summarize each document individually using RAG.
    Returns a dictionary of summaries keyed by document source.
    """

    summaries = {}

    # Create the summarization chain once
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)

    for doc in docs:
        # Determine source for display
        src = doc.metadata.get("source", "Unknown Document")

        try:
            # Run summarization on this document
            summary_result = chain.invoke([doc])
            summaries[src] = summary_result['output_text']
        except Exception as e:
            print(f"Error summarizing {src}: {e}")
            summaries[src] = "Error generating summary."

    return summaries

def run_rag_query(vectordb, query, memory, llm, top_k, fetch_k, sources=None, rerank_top_k=5):
    """Run a RAG query against the vector store."""

    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": top_k, "fetch_k": fetch_k})

    # Apply metadata filter if sources are provided
    if sources:
        retriever_kwargs = {"filter": {"source": {"$in": sources}}}
    else:
        retriever_kwargs = {}

    # ---- Stage 1 of Retrieval: Vector Retrieval (MMR) ----
    docs = retriever.invoke(query, **retriever_kwargs)
    print(f"Retrieved {len(docs)} documents to rerank.")



    # ---- Stage 2 of Retrieval: Cross-Encoder Reranking ----
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    reranked_results = cross_encoder.rank(query=query, documents=[doc.page_content for doc in docs], top_k=rerank_top_k)
    reranked_docs = [docs[item["corpus_id"]] for item in reranked_results]


    # Prepare context from reranked documents and their sources
    context_parts = []
    for doc in reranked_docs:
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