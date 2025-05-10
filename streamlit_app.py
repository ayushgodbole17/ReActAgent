import streamlit as st
from dotenv import load_dotenv
from rag_react_agent import run_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="RAG+ReAct Research Assistant",
    layout="wide",
)

# Sidebar settings
max_steps = st.sidebar.slider(
    "Max reasoning steps", 1, 10, 5
)

# Main UI
st.title("🔍 RAG + ReAct Research Assistant")
st.write(
    "Ask any question about your indexed research papers, "
    "and watch the agent think (Thought), fetch (Action), observe (Observation), and answer (Finish)."
)
question = st.text_input("Enter your question:", "")

if st.button("Run Agent"):
    if not question.strip():
        st.warning("Please enter a question to proceed.")
    else:
        # Setup callback for streaming
        handler = StreamlitCallbackHandler(st.container())
        with st.spinner("Agent is reasoning…"):
            answer = run_agent(
                question,
                max_steps=max_steps,
                callbacks=[handler]
            )
        st.success("✅ Done!")
        st.subheader("Final Answer")
        st.markdown(f"> {answer}")
