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
import uuid

def create_workflow(llm):
    workflow = StateGraph(GraphState)

    # Initialize the graph state
    def init_graph_state():
        return {
            "chat_history": InMemoryChatMessageHistory(),
        }

    workflow.set_initial_state(init_graph_state)

    workflow.add_node("contextualize", contextualize)
    workflow.add_node("extract queries", extract_queries)
    # workflow.add_node("web_search", web_search)
    # workflow.add_node("retrieve", retrieve)
    # workflow.add_node("generate", generate)
    # workflow.add_node("transform_query", transform_query)

    workflow.add_edge(START, "contextualize")
    workflow.add_edge("contextualize", "extract queries")
    workflow.add_edge("extract queries", END)
    # workflow.add_edge("extract queries", "web_search")
    # workflow.add_edge("extract queries", "retrieve")
    # workflow.add_edge(["web_search", "retrieve"], "generate")

    # workflow.add_conditional_edges(
    #     "generate",
    #     grade_generation_v_documents_and_question,
    #     {
    #         "not supported": "generate",
    #         "useful": END,
    #         "not useful": "transform_query",
    #     },
    # )

    # workflow.add_edge("transform_query", "extract queries")

    return workflow.compile()

# Usage
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")

if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY environment variable not set")

# Initialize the ChatMistralAI client with the API key
llm = ChatMistralAI(model="mistral-large-latest", api_key=mistral_api_key)
app = create_workflow(llm)  # Pass your LLM instance here
