import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from config import *

class RAGEngine:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.llm = Ollama(model=LLM_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)

    def process_pdf(self,file_path):
        '''Loads PDF and adds it to persistent collection.'''
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings, 
            collection_name=COLLECTION_NAME,
            client=self.client)
        return vectorstore
    
    def get_qa_chain(self):
        ''' Retrieves the QA chain for the current collection.'''
        #check if collection exists and has items
        try:
            collection = self.client.get_collection(COLLECTION_NAME)
            if collection.count() == 0:
                return None
        except Exception:
            return None     

        vectorstore = Chroma(
            embedding_function=self.embeddings, 
            collection_name=COLLECTION_NAME,
            client=self.client)
        return RetrievalQA.from_chain_type(
            llm=self.llm, 
            chain_type="stuff", 
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        )
    
    def clear_all_data(self):
        '''Deletes the persistent collection and clears the database without breaking the connection'''
        try:
            # delete the specific collection
            self.client.delete_collection(COLLECTION_NAME)
            # recreate an empty collection to maintain the database connection
            self.client.create_collection(COLLECTION_NAME)
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
          

        