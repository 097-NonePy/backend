from langchain.schema import Document
from RAG.tools.vectore_store_retriever import retriever


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    print(state)
    search_query = state["vector_search_query"]

    # Retrieval
    documents = retriever.invoke(search_query)
    return {"vector_search_documents": documents}
