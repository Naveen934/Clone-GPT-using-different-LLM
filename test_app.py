import streamlit as st
import pytest
from app import get_api_key, get_llm_response
from unittest.mock import patch

@pytest.fixture
def setup_session_state():
    """Fixture to reset Streamlit's session state before each test"""
    st.session_state.clear()

# Test model selection and API key input
def test_get_api_key(setup_session_state, monkeypatch):
    # Mock the sidebar text_input to simulate API key entry
    def mock_text_input(label, type=None):
        return "fake-api-key"

    # Replace st.sidebar.text_input with the mock function
    monkeypatch.setattr(st.sidebar, 'text_input', mock_text_input)

    # Test Gemini model
    st.sidebar.selectbox = lambda *args, **kwargs: "Gemini"
    api_key = get_api_key()
    assert api_key == "fake-api-key"
    assert st.session_state["api_key"] == "fake-api-key"

    # Test Mistral or Llama models
    st.sidebar.selectbox = lambda *args, **kwargs: "Llama"
    api_key = get_api_key()
    assert api_key == "fake-api-key"
    assert st.session_state["api_key"] == "fake-api-key"

# Test the LLM interaction with HuggingFace models
@patch("app.invoke_hugging_llm")
def test_llm_response_huggingface(mock_invoke, setup_session_state, monkeypatch):
    # Mock model and API key
    st.sidebar.selectbox = lambda *args, **kwargs: "Llama"
    monkeypatch.setattr(st.session_state, "api_key", "fake-api-key")
    
    # Simulate a HuggingFace response
    mock_invoke.return_value = "This is a test response from Llama."
    
    # Call get_llm_response with the mocked values
    prompt = "What is the meaning of life?"
    response = get_llm_response(st.session_state["api_key"], prompt)
    
    # Check if the HuggingFace API was invoked and returned the correct response
    mock_invoke.assert_called_once()
    assert response == "This is a test response from Llama."

# Test the LLM interaction with Gemini model
@patch("app.ChatGoogleGenerativeAI")
def test_llm_response_gemini(mock_genai, setup_session_state, monkeypatch):
    # Mock model and API key
    st.sidebar.selectbox = lambda *args, **kwargs: "Gemini"
    monkeypatch.setattr(st.session_state, "api_key", "fake-google-api-key")

    # Simulate a Gemini response
    mock_llm_instance = mock_genai.return_value
    mock_llm_instance.invoke.return_value.content = "This is a test response from Gemini."

    # Call get_llm_response with the mocked values
    prompt = "What is the meaning of life?"
    response = get_llm_response(st.session_state["api_key"], prompt)

    # Check if the Google GenAI API was invoked and returned the correct response
    mock_llm_instance.invoke.assert_called_once_with(prompt)
    assert response == "This is a test response from Gemini."

# Test question input and submission (integration-level test)
def test_question_submission(setup_session_state, monkeypatch):
    # Mock text_input and button press
    monkeypatch.setattr(st, 'text_input', lambda label: "What is AI?")
    monkeypatch.setattr(st, 'button', lambda label: True)

    # Mock the response from get_llm_response
    monkeypatch.setattr('app.get_llm_response', lambda api_key, prompt: "AI stands for Artificial Intelligence.")

    # Ensure the correct response is returned when the button is pressed
    question = st.text_input("Ask your question")
    button_pressed = st.button("Submit")

    if button_pressed:
        assert question == "What is AI?"
        response = get_llm_response(st.session_state.get("api_key", ""), "What is AI?")
        assert response == "AI stands for Artificial Intelligence."
