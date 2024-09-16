from RAG.tools.web_search import web_search_tool
from langchain_core.documents import Document

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    search_query = state["web_search_query"]

    # Web search
    docs = web_search_tool.invoke({"query": search_query})
    web_results = [d["content"] for d in docs]
    # web_results = Document(page_content=web_results)

    return {"web_search_documents": web_results}