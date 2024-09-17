from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from RAG.agents.contextualize import contextualizer
from RAG.graph_state import GraphState
import os


def contextualize_question(state):
    print("---CONTEXTUALIZE QUESTION---")

    question = state["question"]
    chat_history = state["chat_history"]
    result = contextualizer.invoke({"input": question, "chat_history": chat_history})

    return {
        "contextualized_question": result.contextualized_question,
    }
    
