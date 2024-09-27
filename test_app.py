import os
import streamlit as st
from unittest import mock
from dotenv import load_dotenv
from app import get_llm_response, invoke_hugging_llm
import pytest
import google.generativeai as genai

# Load environment variables from the .env file
load_dotenv()

# Access the API keys from the .env file
google_api_key = os.getenv('GOOGLE_API_KEY')
huggingface_api_key = os.getenv('HUGGING_FACE_API_KEY')

# Configure the Google Gemini API
genai.configure(api_key=google_api_key)

# Mocking the Session State to include the 'api_key'
@pytest.fixture
def mock_session_state(monkeypatch):
    # Mocking st.session_state to include the api_key from the .env file
    session_state = {'api_key': google_api_key}  # For Google API
    monkeypatch.setattr(st, 'session_state', session_state)

# Mocking the HuggingFace API call
@mock.patch('app.invoke_hugging_llm')
def test_llm_response_huggingface(mock_invoke, mock_session_state):
    mock_invoke.return_value = "This is a mocked response from HuggingFace."
    
    prompt = "What is AI?"
    response = get_llm_response(st.session_state['api_key'], prompt)
    assert response == "This is a mocked response from HuggingFace."

# Mocking the Google Gemini API call
@mock.patch('app.ChatGoogleGenerativeAI')
def test_llm_response_gemini(mock_chat_gemini, mock_session_state):
    mock_instance = mock_chat_gemini.return_value
    mock_instance.invoke.return_value = mock.Mock(content="This is a mocked response from Gemini.")
    
    prompt = "Explain quantum computing."
    response = get_llm_response(st.session_state['api_key'], prompt)
    assert response == "This is a mocked response from Gemini."

# Test the question submission process
@mock.patch('app.get_llm_response')
def test_question_submission(mock_llm_response, mock_session_state):
    mock_llm_response.return_value = "This is a test response."
    question = "What is the future of AI?"
    st.text_input = mock.Mock(return_value=question)
    st.button = mock.Mock(return_value=True)
    
    assert mock_llm_response(st.session_state['api_key'], question) == "This is a test response."
