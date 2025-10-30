from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredWordDocumentLoader, UnstructuredURLLoader, CSVLoader
import os, tempfile

def load_docs(uploaded_files=None, urls=None):
    documents = []
    # load files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            print(f"Loading file: {uploaded_file.name}")
            suffix = Path(uploaded_file.name).suffix.lower()
            
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            # Load based on type
            if suffix == '.pdf':
                loader = PyPDFLoader(tmp_path)
            elif suffix in ['.docx', '.doc']:
                loader = Docx2txtLoader(tmp_path)
            elif suffix == '.txt':
                loader = TextLoader(tmp_path)
            elif suffix == '.csv':
                loader = CSVLoader(tmp_path)
            else:
                continue  # skip unsupported
            
            try:
                loaded_docs = loader.load()
                for d in loaded_docs:
                    d.metadata["source"] = uploaded_file.name  # Add source
                documents.extend(loaded_docs)
            except Exception as e:
                print(f"Error loading {uploaded_file.name}: {e}")
                continue
            finally:
                os.unlink(tmp_path)  # delete temp file

    if urls:
        for url in urls:
            loader = UnstructuredURLLoader(urls=[url])  # single URL
            try:
                url_docs = loader.load()
                for d in url_docs:
                    d.metadata["source"] = url
                documents.extend(url_docs)
            except Exception as e:
                print(f"Error loading {url}: {e}")

    return documents

 