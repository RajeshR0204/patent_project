# import required libraries
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
#from langchain.text_splitter import NLTKTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st
import sys,yaml,Utilities as ut

def get_data(query):
    chat_history = []
    initdict={}
    initdict = ut.get_tokens()
    hf_token = initdict["hf_token"]
    embedding_model_id = initdict["embedding_model"]
    chromadbpath = initdict["chatPDF_chroma_db"]
    llm_repo_id = initdict["llm_repoid"]
    
    
    # We will use HuggingFace embeddings 
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_id)

    #retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 1})
    # load from disk
    db = Chroma(persist_directory=chromadbpath, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 2})

    llm = HuggingFaceHub(huggingfacehub_api_token=hf_token, 
                        repo_id=llm_repo_id, model_kwargs={"temperature":0.2, "max_new_tokens":50})

    # Create the Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever,return_source_documents=True)
    result = qa_chain({'question': query, 'chat_history': chat_history})
    chat_history.append(result)
    print('Answer: ' + result['answer'] + '\n')
    print (result)
    return result['answer']
    
st.title("PatentGuru Document Reader")

# Main chat form
with st.form("chat_form"):
    query = st.text_input("Chat with PDF: ")
    clear_history = st.checkbox('Clear Chat History') 
    submit_button = st.form_submit_button("Send")    

if submit_button:
    if clear_history:
        st.write("Cleared previous chat history")
    
    response = get_data(query)
    if len(response)>0: 
        response  = str(response.partition("Answer: ")[-1])
    else: response = "No results"
    
    # write results
    st.write (response)


