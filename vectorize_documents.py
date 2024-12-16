import os
import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Ensure that models are loaded locally
os.environ["TRANSFORMERS_CACHE"] = "./transformers_cache"
os.environ["HF_HOME"] = "./huggingface_cache"

# Initialize sentence transformer model and save locally
sentence_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
sentence_model = SentenceTransformer(sentence_model_name)
sentence_model.save('./local_model')

# Load HuggingFace embedding model (Sentence Transformer)
embedding_model = HuggingFaceEmbeddings(model_name="./local_model")

print(dir(embedding_model))

# Function to load documents (PDFs)
def load_documents(directory="data", file_pattern="*.pdf"):
    loader = DirectoryLoader(
        path=directory,
        glob=file_pattern,
        loader_cls=PyMuPDFLoader
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} PDF documents from {directory}.")
    return docs

# Function to load CSV files
def load_csv(directory="data", file_pattern="*.csv"):
    csv_files = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, file))
            # Convert each row into a document
            for _, row in df.iterrows():
                csv_files.append({"text": str(row.to_dict())})
    print(f"Loaded {len(csv_files)} CSV rows from {directory}.")
    return csv_files

# Function to load text files
def load_text(directory="data", file_pattern="*.txt"):
    text_files = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            with open(os.path.join(directory, file), 'r') as file_content:
                text_files.append({"text": file_content.read()})
    print(f"Loaded {len(text_files)} text files from {directory}.")
    return text_files

# Function to process image files and generate embeddings using CLIP model
def load_images(directory="data", file_pattern="*.jpg"):
    image_files = []
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir="./transformers_cache")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir="./transformers_cache")
    
    for file in os.listdir(directory):
        if file.endswith(".jpg") or file.endswith(".png"):
            image = Image.open(os.path.join(directory, file)).convert("RGB")
            # Preprocess image and get embeddings
            inputs = clip_processor(images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                image_embeddings = clip_model.get_image_features(**inputs)
            image_files.append({"text": image_embeddings.squeeze().numpy()})
    
    print(f"Loaded {len(image_files)} images from {directory}.")
    return image_files

# Loading and splitting documents
def split_text_into_chunks(docs, chunk_size=4000, overlap=500):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

# Load different types of documents
documents = load_documents()  # Loading PDF documents
csv_docs = load_csv()         # Loading CSV documents
text_docs = load_text()       # Loading Text files
image_docs = load_images()    # Loading Images

# Combine all documents
all_documents = documents + csv_docs + text_docs + image_docs
print(f"Total documents: {len(all_documents)}")

# Split text into chunks
chunks = split_text_into_chunks(all_documents)

# Initialize and persist the vector database
persist_dir = "vector_database"
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

# Create or load Chroma collection
vector_store = Chroma.from_documents(
    documents=all_documents,
    embedding=embedding_model,
    collection_name="my_collection",
    collection_metadata={"dimensionality": 768}  # Explicitly set to match your model
)

print("Documents have been successfully vectorized.")
print(f"Total chunks in vector store: {len(chunks)}")
