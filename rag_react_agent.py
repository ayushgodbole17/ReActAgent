# rag_react_agent.py
import os
import logging
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Use community integrations for up-to-date compatibility
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.callbacks.manager import CallbackManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1️⃣ Initialize embeddings and vector store
embeddings = OpenAIEmbeddings()
store = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)

# 2️⃣ RAG retrieval with caching and error handling
top_k = int(os.getenv("RAG_TOP_K", 5))

@lru_cache(maxsize=128)
def rag_retrieve(query: str) -> str:
    """
    Retrieve top-k document chunks for the given query, cached for efficiency.
    """
    try:
        docs = store.similarity_search(query, k=top_k)
        return "\n\n".join(d.page_content for d in docs)
    except Exception as e:
        logger.error(f"RAG retrieval failed for query '{query}': {e}")
        return ""  # return empty result on failure

rag_tool = Tool(
    name="RAGRetrieve",
    func=rag_retrieve,
    description="Retrieve relevant document snippets for a given query using cached similarity search."
)

# 3️⃣ Prompt template: include explicit Final Answer marker
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template="""
Answer the following question as best you can. You have access to the following tool:

{tools}

Follow this exact format (no extra text):

Question: the input question
Thought: think about what to do next
Action: one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed) ...
Thought: I now know the final answer
Final Answer: the final answer

Begin!
Question: {input}
{agent_scratchpad}
"""
)

# 4️⃣ Agent execution: streaming via callbacks, non-streaming returns trace
def run_agent(
    question: str,
    max_steps: int = 5,
    callbacks: list = None
):
    """
    Execute the RAG+ReAct agent.
    - Streams intermediate steps if callbacks provided.
    - Returns the final answer (and trace tuple if non-streaming).
    """
    # Choose model settings
    if callbacks:
        cb_manager = CallbackManager(callbacks)
        llm = ChatOpenAI(
            model=os.getenv("LLM_INTERMEDIATE_MODEL", "gpt-3.5-turbo"),
            temperature=0.2,
            streaming=True,
            callback_manager=cb_manager
        )
        return_intermediate = False
    else:
        llm = ChatOpenAI(
            model=os.getenv("LLM_FINAL_MODEL", "gpt-4"),
            temperature=0.2
        )
        return_intermediate = True

    # Create agent and executor
    agent = create_react_agent(llm, [rag_tool], prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=[rag_tool],
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=return_intermediate,
        max_iterations=max_steps
    )

    # Always invoke to fire callbacks correctly
    result = executor.invoke({"input": question})

    if callbacks:
        # Streaming mode: just return the final output
        return result.get("output")
    # Non-streaming: return both answer and trace
    return result.get("output"), result.get("intermediate_steps", [])
