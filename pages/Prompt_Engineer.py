
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate
from langchain_community.llms import LlamaCpp
#from langchain.chains import RetrievalQA
#from langchain_community.embeddings import SentenceTransformerEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

#from langchain.schema import HumanMessage

import os
import json,streamlit as st
from pathlib import Path

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # "/" is a marker to show difference 
        # you don't need it 
        #self.text+=token+"/" 
        self.text+=token
        self.container.markdown(self.text) 

st.title("Prompt Engineer")

# Main chat form
with st.form("chat_form"):
    query = st.text_input("Enter the topic you want to generate prompt for?: ")
    #LLM_Summary = st.checkbox('Summarize results with LLM') 
    submit_button = st.form_submit_button("Send")    

     
    template = """
    <s>[INST] <<SYS>>
    Act as a patent advisor by providing subject matter expertise on any topic. Provide detailed and elaborate answers
    <</SYS>>

    {text} [/INST]
    """
    response=""
    prompt = PromptTemplate(
        input_variables=["text"],
        template=template,
    )
    text = "Help me create a good prompt for the following: Information that is needed to file a US patent application for " + query
    #print(prompt.format(text=query))

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    #model_path = "C:\Rajesh\AI-ML-Training\LLM\llama-2-7b.Q4_K_M.gguf"\
    model_path = "C:\Rajesh\AI-ML-Training\LLM\zephyr-7b-beta.Q5_K_S.gguf"
    chat_box=st.empty() 
    stream_handler = StreamHandler(chat_box)
    
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.8,
        max_tokens=500,
        top_p=1,
        #streaming=True,
        #callback_manager=callback_manager,
        callback_manager = [stream_handler],
        verbose=True,  # Verbose is required to pass to the callback manager
    )

if submit_button:
    #st.write("Fetching results..\n")
    output = llm.invoke(prompt.format(text=text))
    #response = response+output
    #st.write(response)
    #response = output([HumanMessage(content=query)])    
    #llm_response = output.content
    #st.markdown(output)




    
    
