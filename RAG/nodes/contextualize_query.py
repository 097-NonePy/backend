from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from RAG.agents.contextualize import contextualizer
from RAG.graph_state import GraphState
import os

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

def contextualize_question(state):
    question = state["question"]
    # chat_history = state["chat_history"]
    chat_history = GraphState.get_state({"configurable": {"thread_id": "1"}}).chat_history
    print(chat_history)
    result = contextualizer.invoke({"input": question, "chat_history": chat_history})
    print("contextualised result", result.contextualized_question)
    chat_history.append(HumanMessage(content=result.contextualized_question))
    state["chat_history"] = chat_history

    return {
        "contextualized_question": result,
        "question": question,
        "state": state
    }
    
