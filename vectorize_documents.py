from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# Initialize HuggingFace embedding model
embedding_model = HuggingFaceEmbeddings()

# Define a function to load documents from a directory
def load_documents(directory="data", file_pattern="./*.pdf"):
    loader = DirectoryLoader(
        path=directory,
        glob=file_pattern,
        loader_cls=UnstructuredFileLoader
    )
    return loader.load()

# Define a function to split text into chunks
def split_text_into_chunks(docs, chunk_size=4000, overlap=500):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_documents(docs)

# Load and split documents
documents = load_documents()
chunks = split_text_into_chunks(documents)

# Initialize and persist the vector database
persist_dir = "vector_database"
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)  # Ensure the directory exists

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=persist_dir
)

print("Documents have been successfully vectorized.")
