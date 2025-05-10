import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Community packages (latest LangChain v0.2+)
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.callbacks.manager import CallbackManager

# 1) Initialize embeddings and vector store
embeddings = OpenAIEmbeddings()
store = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)

# 2) Define RAG retrieval function and wrap as a Tool
def rag_retrieve(query: str, k: int = 5) -> str:
    docs = store.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)

rag_tool = Tool(
    name="RAGRetrieve",
    func=rag_retrieve,
    description="Retrieve relevant document chunks for a given query"
)

# 3) Prompt template for the ReAct agent
template = """
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
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template=template
)

# 4) Agent execution function
def run_agent(
    question: str,
    max_steps: int = 5,
    callbacks: list = None
):
    """
    Runs the ReAct agent with RAG retrieval.
    - If callbacks provided, stream intermediate steps via those handlers.
    - Returns final answer (and trace when not streaming).
    """
    # Setup LLM with or without streaming
    if callbacks:
        cb_manager = CallbackManager(callbacks)
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.2,
            streaming=True,
            callback_manager=cb_manager
        )
        return_intermediate = False
    else:
        llm = ChatOpenAI(
            model="gpt-4",
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

    # Always use invoke so streaming callbacks get fired, and output key exists
    result = executor.invoke({"input": question})

    # Return based on mode
    if callbacks:
        # Streaming: result['output'] has the final answer
        return result.get("output")
    # Non-streaming: also return trace
    return result.get("output"), result.get("intermediate_steps", [])
