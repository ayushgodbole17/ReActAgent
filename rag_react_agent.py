# rag_react_agent.py

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from openai_utils import chat_completion

# Load environment variables
load_dotenv()

# Initialize persistent Chroma vector store
store = Chroma(persist_directory="vector_db", embedding_function=None)

def rag_retrieve(query: str, k: int = 5) -> str:
    """
    Retrieve the top-k most relevant document chunks for the query.
    """
    docs = store.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)

def parse_react(output: str) -> tuple[str, str]:
    """
    Extract the latest Thought and Action lines from the LLM output.
    Raises ValueError if parsing fails.
    """
    thought = None
    action = None
    for line in output.splitlines():
        if line.startswith("Thought"):
            thought = line.split(":", 1)[1].strip()
        if line.startswith("Action"):
            action = line.split(":", 1)[1].strip()
            break
    if thought is None or action is None:
        raise ValueError(f"Could not parse Thought/Action from output:\n{output}")
    return thought, action

# … keep your imports, load_dotenv(), and rag_retrieve/parse_react definitions …

def react_agent(question: str, max_steps: int = 5) -> str:
    """
    Run the ReAct loop: use RAGRetrieve[...] for retrieval and Finish[...] for final answer.
    If the model fails to follow ReAct format, return its raw output as the answer.
    """
    # 1. Stronger system instruction to enforce ReAct formatting
    system_msg = {
        "role": "system",
        "content": (
            "You are a research assistant. "
            "Answer every question using exactly the following ReAct format:\n"
            "Thought 1: ...\n"
            "Action 1: RAGRetrieve[<query>]\n"
            "Observation 1: ...\n"
            "... (repeat Thought/Action/Observation as needed) ...\n"
            "Finish[<final answer>]\n"
            "Do not output anything outside these labels."
        )
    }
    context = [f"Question: {question}"]

    for i in range(1, max_steps + 1):
        # Build the chat prompt
        user_content = "\n".join(context) + f"\nThought {i}:"
        messages = [system_msg, {"role": "user", "content": user_content}]

        # Get model output
        output = chat_completion(messages).strip()

        # Try parsing Thought/Action
        try:
            thought, action = parse_react(output)
        except ValueError:
            # Model didn't follow ReAct format: treat raw output as the final answer
            return output

        # Handle retrieval vs. finish
        if action.startswith("RAGRetrieve["):
            query = action[len("RAGRetrieve["):-1]
            observation = rag_retrieve(query)
            context.extend([
                f"Thought {i}: {thought}",
                f"Action {i}: {action}",
                f"Observation {i}: {observation}"
            ])
        elif action.startswith("Finish["):
            return action[len("Finish["):-1]
        else:
            # Unexpected label—abort with raw output
            return output

    return "Unable to answer within step limit."


if __name__ == "__main__":
    question = "Explain the key innovation of the Transformer architecture."
    answer = react_agent(question)
    print("Answer:", answer)
