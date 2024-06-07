import streamlit as st
import google.generativeai as genai
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os

st.set_page_config(page_title="AI Tool", page_icon=":robot:")
st.title("Gpt clone")
st.sidebar.title("select your LLM model")

model=st.sidebar.selectbox("please select any models ",("Gemini","mistal","llama"), index=None,
   placeholder="Select your LLM model...")
st.write("Your LLM Model is:",model)

if "api_key" not in st.session_state:
    st.session_state["api_key"]=''

def get_api():
    if model=="Gemini":
        st.session_state["api_key"]=st.sidebar.text_input("please ender your Gemini API",type='password')
    else:
        st.session_state["api_key"]=st.sidebar.text_input("please ender your HuggingFace API",type='password')
    return st.session_state["api_key"]



def hugging_llm(model,api):
    os.environ["HUGGINGFACEHUB_API_TOKEN"]=api
    llm=HuggingFaceEndpoint(repo_id=model)
    response=llm.invoke(prompt)
    return response


def get_answer(api_key):
    if model=="mistal":
        model_name="mistralai/Mistral-7B-Instruct-v0.3"
        response=hugging_llm(model_name,api_key)      
    elif model=="llama":
        model_name="meta-llama/Meta-Llama-3-8B-Instruct"
        response=hugging_llm(model_name,api_key)
    elif model=="Gemini":
        st.session_state["api_key"]=st.sidebar.text_input("please ender your Google API",type='password')
        os.environ['GOOGLE_API_KEY']=api_key
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        llm=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
        response=llm.invoke(prompt).content

    return response


api=get_api()
if api:
  st.write("getting API")
  question=st.text_input("Ask you questions")
  button2=st.button("summit")

from langchain_community.tools import DuckDuckGoSearchRun
searchduck= DuckDuckGoSearchRun()
search=searchduck.run(question)

template="You are AI assitance give related answer to users question, that question is {question} if user ask question from current affair use the duckduck go search result as content, that duckduck go search is {search}"
example_promt=PromptTemplate(input_variables=["question","search"],template=template)
prompt=example_promt.format(question=question,search=search)

if button2:    
    response=get_answer(st.session_state["api_key"])
    st.write(response)





