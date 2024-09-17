from langgraph.graph import END, StateGraph, START
from RAG.graph_state import GraphState
from RAG.nodes.extract_queries import extract_queries
from RAG.nodes.web_search import web_search
from RAG.nodes.retrieve import retrieve
from RAG.nodes.generate import generate
from RAG.nodes.transform import transform_query
from RAG.nodes.contextualize_query import contextualize_question as contextualize
from RAG.edges.generation_grader import grade_generation_v_documents_and_question
from langchain_core.chat_history import InMemoryChatMessageHistory


import os
# import uuid
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

workflow = StateGraph(GraphState)

workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("contextualize", contextualize)
workflow.add_node("extract queries", extract_queries)

workflow.add_edge(START, "contextualize")
workflow.add_edge("contextualize", "extract queries")
workflow.add_edge("extract queries", "web_search")
workflow.add_edge("extract queries", "retrieve")

workflow.add_edge(["web_search", "retrieve"], "generate")

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

workflow.add_edge("transform_query", "extract queries")

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

try:
    graph_image = app.get_graph(xray=True).draw_mermaid_png()
    with open("graph_image.png", "wb") as f:
        f.write(graph_image)
except Exception:
    pass
  