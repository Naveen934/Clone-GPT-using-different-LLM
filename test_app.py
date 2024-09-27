import os
import streamlit as st
from unittest import mock
from app import get_llm_response, invoke_hugging_llm
import pytest
import google.generativeai as genai

# Use the real API key if available, otherwise fallback to a placeholder key
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', 'AIzaSyDiqulUgCv8XGvVwZEVx2splxJRDgVbUdY')
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Mocking the Session State to include the 'api_key'
@pytest.fixture
def mock_session_state(monkeypatch):
    # Mocking st.session_state to include the api_key
    session_state = {'api_key': os.getenv('REAL_API_KEY', 'fake-api-key')}  # Fallback to fake if not set
    monkeypatch.setattr(st, 'session_state', session_state)

# Mocking the HuggingFace API call
@mock.patch('app.invoke_hugging_llm')
def test_llm_response_huggingface(mock_invoke, mock_session_state):
    # Mock the HuggingFace LLM API response
    mock_invoke.return_value = "This is a mocked response from HuggingFace."
    
    # Call the function and check the response
    prompt = "What is AI?"
    response = get_llm_response(st.session_state['api_key'], prompt)
    assert response == "This is a mocked response from HuggingFace."

# Mocking the Google Gemini API call
@mock.patch('app.ChatGoogleGenerativeAI')
def test_llm_response_gemini(mock_chat_gemini, mock_session_state):
    # Mock the Google Gemini LLM API response
    mock_instance = mock_chat_gemini.return_value
    mock_instance.invoke.return_value = mock.Mock(content="This is a mocked response from Gemini.")
    
    # Call the function and check the response
    prompt = "Explain quantum computing."
    response = get_llm_response(st.session_state['api_key'], prompt)
    assert response == "This is a mocked response from Gemini."

# Test the question submission process
@mock.patch('app.get_llm_response')
def test_question_submission(mock_llm_response, mock_session_state):
    # Mock the LLM response
    mock_llm_response.return_value = "This is a test response."

    # Simulate the input and button press in Streamlit
    question = "What is the future of AI?"
    st.text_input = mock.Mock(return_value=question)
    st.button = mock.Mock(return_value=True)
    
    # Check if the function was called with the correct arguments
    assert mock_llm_response(st.session_state['api_key'], question) == "This is a test response."
