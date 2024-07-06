# import required libraries
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain import PromptTemplate

import streamlit as st
import sys,yaml
import chromadb
import Utilities as ut

hf_token=""
chromadbpath=""
chromadbcollname=""
embedding_model_id=""
llm_repo_id=""
#embeddings=None
#chroma_client=None


def filterdistance(distcoll):
    myemptydict={}
    if len(distcoll) < 0:myemptydict
    for distances in distcoll['distances']:
        for distance in distances:
            if distance<50: return distcoll
            else: return myemptydict
           
def get_collections(query):
    #myemptydict={}
    result=""
    initdict={}
    initdict = ut.get_tokens()
    hf_token = initdict["hf_token"]
    embedding_model_id = initdict["embedding_model"]
    chromadbpath = initdict["dataset_chroma_db"]
    chromadbcollname = initdict["dataset_chroma_db_collection_name"]
    llm_repo_id = initdict["llm_repoid"]
    
    embedding_model = SentenceTransformer(embedding_model_id)
    #print(chromadbpath)
    #print(chromadbcollname)
    chroma_client = chromadb.PersistentClient(path = chromadbpath)
    collection = chroma_client.get_collection(name = chromadbcollname)
 
    #collection = chroma_client.get_or_create_collection(name=chromadbcollname)
    query_vector = embedding_model.encode(query).tolist()
    output = collection.query(
        query_embeddings=[query_vector],
        n_results=1,
        #where={"distances": "is_less_than_1"},
        include=['documents','distances'],
        
        )
    #Filter for distances
    output = filterdistance(output)
    
    if len(output)>0: 
        template = """
        <s>[INST] <<SYS>>
        Act as a patent assistant who is helping summarize and neatly format the results for better readability. Ensure the output is gramatically correct and easily understandable 
        <</SYS>>

        {text} [/INST]
        """
        #Build the prompt template
        prompt = PromptTemplate(
            input_variables=["text"],
            template=template,
        )
        text = output
        
        llm = HuggingFaceHub(huggingfacehub_api_token=hf_token, 
                        repo_id=llm_repo_id, model_kwargs={"temperature":0.2, "max_new_tokens":50})

        result = llm.invoke(prompt.format(text=text))
        print (result)
    return result
  
    return output
    # extract and apply distance condition

st.title("BIG Patent Search")

# Main chat form
with st.form("chat_form"):
    query = st.text_input("Enter the abstract search for similar patents: ")
    #LLM_Summary = st.checkbox('Summarize results with LLM') 
    submit_button = st.form_submit_button("Send")    

if submit_button:
    st.write("Fetching results..\n")
    results =  get_collections(query)
    
    if len(results)>0: 
        #docids = results["documents"]
        response = "There are existing patents related to -    "
        substring = results.partition("[/ASSistant]")[-1]
        if len(substring)>0:
            response  = response + str(substring)
        else: 
            response = response + results.partition("[/INST]")[-1]
            
    else: response = "No results"
    
    st.write (response)
    
