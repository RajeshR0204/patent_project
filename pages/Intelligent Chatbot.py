from langchain_community.llms import HuggingFaceEndpoint
import streamlit as st, Utilities as ut
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_community.chat_models.huggingface import ChatHuggingFace
#from langchain_openai import OpenAI

from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

st_callback = StreamlitCallbackHandler(st.container())

initdict={}
initdict = ut.get_tokens()
hf_token = initdict["hf_token"]
reactstyle_prompt = initdict["reactstyle_prompt"]
serpapi_api_key = initdict["serpapi_api_key"]
llm_repoid = initdict["llm_repoid"]

llm = HuggingFaceEndpoint(repo_id=llm_repoid,huggingfacehub_api_token=hf_token,temperature=0.9,verbose=True)

tools = load_tools(["serpapi"],llm=llm,serpapi_api_key=serpapi_api_key)   
prompt = hub.pull(reactstyle_prompt)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)

chat_model = ChatHuggingFace(llm=llm)
chat_model_with_stop = chat_model.bind(stop=["\nObservation"])

st.title("PatentGuru - Intelligent Chatbot")

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        
        response = agent_executor.invoke(
            {"input": prompt}, {"callbacks": [st_callback], "handle_parsing_errors":True}
        )
        st.write(response["output"])