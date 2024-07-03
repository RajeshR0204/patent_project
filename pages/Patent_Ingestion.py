# import required libraries
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
import tensorflow_datasets as tfds
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer
import textwrap
import chromadb
import streamlit as st
import sys,yaml
import uuid
import Utilities as ut


def text_summarizer(text):
    initdict = ut.get_tokens()
    BART_Model_Name = initdict["BART_model"]
    #model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(BART_Model_Name)
    tokenizer = BartTokenizer.from_pretrained(BART_Model_Name)

    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    formatted_summary = "\n".join(textwrap.wrap(summary, width=80))
    
    return formatted_summary

def load_patentBIGdata():
    
    initdict={}
    initdict = ut.get_tokens()
    
    embedding_model_id = initdict["embedding_model"]
    chromadbpath = initdict["dataset_chroma_db"]
    chromadbcollname = initdict["dataset_chroma_db_collection_name"]
       
    embedding_model = SentenceTransformer(embedding_model_id)
    
    chroma_client = chromadb.PersistentClient(path= chromadbpath)

    collection = chroma_client.get_or_create_collection(name=chromadbcollname)

    
    # Load the Big patent dataset
    ds = load_dataset("big_patent", "a", split="validation[:1%]",trust_remote_code=True)

    for record in ds.take(10):
        abstract, desc = record ["abstract"], record["description"]
        # Summarize to 150 words
        abstract = text_summarizer(abstract)
        textembeddings = embedding_model.encode(abstract).tolist()
        
        genguid=str(uuid.uuid4())
        #take 8 characters
        uniqueid = genguid[:8]
        # Now we will store the expert explanation field of first 10 questions from dataset into collection. 
        collection.add(
            documents=[
                abstract
            ],
            embeddings=[textembeddings],
            ids=[uniqueid]
        )
        #print(abstract)
    
st.title("Patent Ingestion - BIG Patent")

# Main chat form
with st.form("chat_form"):
   
    submit_button = st.form_submit_button("Upload BIG Patent data...")    

if submit_button:
    load_patentBIGdata()
    response = "BIG Patent dataset was successfully loaded"

    st.write (response)
    
    