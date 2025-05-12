import streamlit as st
# Must set config before any other Streamlit commands
st.set_page_config(page_title="RAG+ReAct Research Assistant", layout="wide")

from dotenv import load_dotenv
import os
from typing import List, Tuple
from rag_react_agent import run_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Load env vars
load_dotenv()

# Initialize session state for conversation history
if "history" not in st.session_state:
    st.session_state.history: List[Tuple[str, str]] = []

# Sidebar: file uploader for new documents
st.sidebar.header("Document Uploader")
uploader = st.sidebar.file_uploader("Upload PDF/TXT to index", accept_multiple_files=True)
if uploader:
    for file in uploader:
        # Save to local directory and re-index (stub)
        with open(os.path.join("vector_db", file.name), "wb") as f:
            f.write(file.getbuffer())
    st.sidebar.success("Uploaded and added to index. Re-run to include new docs.")

# Sidebar settings
st.sidebar.header("Settings")
max_steps = st.sidebar.slider("Max reasoning steps", 1, 10, 5)

# Main UI
st.title("🔍 RAG + ReAct Research Assistant")
st.write(
    "Ask any question about your indexed research papers, "
    "and watch the agent think (Thought), fetch (Action), observe (Observation), and answer (Finish)."
)

# Render past messages
for q, a in st.session_state.history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)

# User input
question = st.chat_input("Ask a question about your papers...")
if question:
    # Display user message
    st.chat_message("user").write(question)

    # Run agent with streaming
    handler = StreamlitCallbackHandler(st.container())
    with st.spinner("Agent is reasoning..."):
        answer = run_agent(question, max_steps=max_steps, callbacks=[handler])

    # Display final answer
    st.chat_message("assistant").write(answer)

    # Update history
    st.session_state.history.append((question, answer))
