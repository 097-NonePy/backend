from langgraph.graph import END, StateGraph, START

from RAG.graph_state import GraphState

from RAG.nodes.extract_queries import extract_queries
from RAG.nodes.web_search import web_search
from RAG.nodes.retrieve import retrieve
from RAG.nodes.compare_generate import generate
from RAG.nodes.transform import transform_query


from RAG.edges.generation_grader import grade_generation_v_documents_and_question

workflow = StateGraph(GraphState)

workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("extract queries", extract_queries)

workflow.add_edge(START, "extract queries")
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

app = workflow.compile()

try:
    graph_image = app.get_graph(xray=True).draw_mermaid_png()
    with open("compare_graph.png", "wb") as f:
        f.write(graph_image)
except Exception:
    # This requires some extra dependencies and is optional
    pass
