import streamlit as st
import os
from rag import RAGEngine

st.set_page_config(page_title="Personal RAG Assistant", layout="wide")
st.title("📚 Personal RAG Assistant")

# Initialize RAG Engine
if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine()

engine = st.session_state.engine

with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    if st.button("Index Document") and uploaded_file:
        with st.spinner("Processing document..."):
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            engine.process_pdf(file_path)
            os.remove(file_path)
            st.success("Document indexed successfully!")
    
    st.divider()

    if st.button("Clear Collection", type = "primary"):
        if engine.clear_all_data():
            st.session_state.messages = []
            st.success("Collection cleared successfully!")
            st.rerun()

#--- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        chain = engine.get_qa_chain()
        if chain:
            with st.spinner("Generating response..."):
                response = chain.invoke(prompt)
                st.markdown(response["result"])
                st.session_state.messages.append({"role": "assistant", "content": response["result"]})
        else:
            st.warning("No documents found in the database. Please upload and index a document first.")
            st.session_state.messages.append({"role": "assistant", "content": "No documents found in the collection. Please upload and index a document first."})
        

