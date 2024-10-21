import os
import json

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # Ensure you have the correct import here
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from vectorize_documents import embedding_model  # Ensure embedding_model is correctly defined

os.environ["TRANSFORMERS_OFFLINE"] = "1"

working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_database"
    # Use the pre-initialized embedding model directly
    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embedding_model)
    return vectorstore

def chat_chain(vectorstore):
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=True,
        return_source_documents=True
    )

    return chain

st.set_page_config(
    page_title="Chatbot",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversational_chain" not in st.session_state:  # Corrected typo here
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask AIDA...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Handle response generation
        try:
            response = st.session_state.conversational_chain({"question": user_input})
            retrieved_docs = response.get('source_documents', [])
                
            # Print or log the retrieved documents for debugging
            print("Retrieved documents:", retrieved_docs)            
            assistant_response = response["answer"]
            st.markdown(assistant_response)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
        except Exception as e:
            st.markdown("Sorry, there was an error processing your request.")
            print(f"Error: {e}")  # Log the error for debugging
