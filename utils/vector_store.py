from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS

def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    """Chunk documents into smaller pieces."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def create_vectorstore(documents):
    """Create a Chroma vector store from documents."""
    if not documents:
        raise ValueError("No documents found. Please upload or enter valid files/URLs.")
    
    # Chunk documents
    chunked_docs = chunk_documents(documents)

    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create Chroma vector store
    vectordb = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        collection_name="user_docs",
    )
    
    return vectordb