from langchain_community.llms import HuggingFaceEndpoint
import streamlit as st, Utilities as ut
from langchain import hub, SerpAPIWrapper
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType


from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

st_callback = StreamlitCallbackHandler(st.container())

initdict={}
initdict = ut.get_tokens()
hf_token = initdict["hf_token"]
reactstyle_prompt = initdict["reactstyle_prompt"]
tavily_api_key = initdict["tavily_api_key"]
llm_repoid = initdict["llm_repoid"]

llm = HuggingFaceEndpoint(repo_id=llm_repoid,huggingfacehub_api_token=hf_token,temperature=0.9)
from tavily import TavilyClient
tavily = TavilyClient(api_key=tavily_api_key)




st.title("PatentGuru - Intelligent Chatbot")

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        
        #response = agent_executor.invoke(
        #    {"input": prompt}, {"callbacks": [st_callback], "handle_parsing_errors":True}
        #)
        # response = self_ask_with_search.run(prompt)
        response = tavily.qna_search(query=prompt)
        st.write(response)