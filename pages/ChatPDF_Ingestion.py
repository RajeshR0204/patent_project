# File Selection Drop Down
import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys,yaml,Utilities as ut


st.set_page_config(page_title="ChatPDF Ingestion", page_icon="📈")

def load_pdf():
   
   # Load the pdf file and split it into smaller chunks
   initdict={}
   initdict = ut.get_tokens()
   hf_token = initdict["hf_token"]
   embedding_model_id = initdict["embedding_model"]
   chromadbpath = initdict["chatPDF_chroma_db"]
   
   embeddings = HuggingFaceEmbeddings(model_name=embedding_model_id)
   
   loader = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)

   documents = loader.load()
   #print (len(documents))
   
   # Split the documents into smaller chunks 

   text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
   texts = text_splitter.split_documents(documents)
    
   #Using Chroma vector database to store and retrieve embeddings of our text
   db = Chroma.from_documents(texts, embeddings, persist_directory=chromadbpath)
   return db

st.title("PatentGuru  - Document Ingestion ")
# Main chat form
with st.form("chat_form"):
    #query = st.text_input("You: ")
    submit_button = st.form_submit_button("Upload..")    

if submit_button:
    load_pdf()
        
    st.write ("Uploaded successfully")