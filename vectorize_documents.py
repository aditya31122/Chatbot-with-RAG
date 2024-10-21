import os
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
model.save('local_model')


# Initialize HuggingFace embedding model
embedding_model = HuggingFaceEmbeddings(model_name="local_model")

print(embedding_model.client.get_sentence_embedding_dimension())

# Load documents with PyMuPDFLoader to support text and images
def load_documents(directory="data", file_pattern="./*.pdf"):
    loader = DirectoryLoader(
        path=directory,
        glob=file_pattern,
        loader_cls=PyMuPDFLoader
    )
    docs=loader.load()
    print(f"Loaded {len(docs)} documents from {directory}.")
    return docs

# Load and split documents
documents = load_documents()  # Load all documents
print(f"Number of documents loaded: {len(documents)}")  # Print the count of documents

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
    os.makedirs(persist_dir)

# Create or load Chroma collection
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection",
    collection_metadata={"dimensionality": 768}  # Explicitly set to match your model

)


print("Documents have been successfully vectorized.")
print(f"Total chunks in vector store: {len(chunks)}")
