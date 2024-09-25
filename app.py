import streamlit as st
import google.generativeai as genai
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os

# Streamlit page configuration
st.set_page_config(page_title="AI Tool", page_icon=":robot:")
st.title("GPT Clone")
st.sidebar.title("Select your LLM Model")

# Sidebar to select the model
model = st.sidebar.selectbox("Please select any model:", 
                             ("Gemini", "Mistral", "Llama"), 
                             placeholder="Select your LLM model...")

st.write("Your LLM Model is:", model)

# Initialize API key state
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ''

# Function to get API key input from the user
def get_api_key():
    if model == "Gemini":
        st.session_state["api_key"] = st.sidebar.text_input("Please enter your Gemini API key", type='password')
    else:
        st.session_state["api_key"] = st.sidebar.text_input("Please enter your HuggingFace API key", type='password')
    return st.session_state["api_key"]

# Function to interact with HuggingFace models
def invoke_hugging_llm(model_name, api_key):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    llm = HuggingFaceEndpoint(repo_id=model_name)
    response = llm.invoke(prompt)
    return response

# Function to get the model's response based on the selected LLM
def get_llm_response(api_key):
    if model == "Mistral":
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        response = invoke_hugging_llm(model_name, api_key)
    elif model == "Llama":
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        response = invoke_hugging_llm(model_name, api_key)
    elif model == "Gemini":
        os.environ['GOOGLE_API_KEY'] = api_key
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        response = llm.invoke(prompt).content
    return response

# Get API key
api_key = get_api_key()

# Display success message if API key is provided
if api_key:
    st.success("API Key Acquired")
    
    # Text input for user's question
    question = st.text_input("Ask your question")
    
    # Button to submit the question
    button2 = st.button("Submit")

    # Search tool integration (DuckDuckGo)
    from phi.assistant import Assistant
    from phi.tools.duckduckgo import DuckDuckGo

    # Initialize the search tool
    search_tool = Assistant(tools=[DuckDuckGo()], show_tool_calls=True)

    # Check if a question has been entered
    if question:
        search_result = search_tool.run(question)  # Adjusted from 'print_response' to 'run'

    # Create the prompt using the search result
    template = """You are an AI assistant. Provide relevant answers to the user's question. 
                The user's question is: {question}. 
                If the user asks about current affairs, use the DuckDuckGo search result as context. 
                The DuckDuckGo search result is: {search}"""

    example_prompt = PromptTemplate(input_variables=["question", "search"], template=template)
    prompt = example_prompt.format(question=question, search=search_result)

        # If the submit button is clicked, get the response from the selected model
    if button2:
        response = get_llm_response(st.session_state["api_key"])
        st.write(response)
