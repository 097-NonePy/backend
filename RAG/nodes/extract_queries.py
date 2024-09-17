from RAG.agents.extractor import question_extractor
from langchain_core.messages import HumanMessage

def extract_queries(state):
    from RAG.graph import app
    """
    Extract queries for vector search and web search.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---EXTRACT QUERIES---")

    contextualized_question = state["contextualized_question"]
    source = question_extractor.invoke({"question": contextualized_question})
    
    return {
        "namal_vector_search_query": source.namal_vector_search_query, 
        "ranil_vector_search_query": source.ranil_vector_search_query,
        "sajith_vector_search_query": source.sajith_vector_search_query,
        "web_search_query": source.web_search_query,
        "question": contextualized_question,
        }
