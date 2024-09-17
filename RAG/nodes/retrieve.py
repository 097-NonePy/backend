from langchain.schema import Document
from RAG.tools.vectore_store_retriever import sajith_retriever, namal_retriever, ranil_retriever


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")

    namal_vector_search_query = state["namal_vector_search_query"]
    ranil_vector_search_query = state["ranil_vector_search_query"]
    sajith_vector_search_query = state["sajith_vector_search_query"]

    # Retrieval
    namal_documents = namal_retriever.get_relevant_documents(namal_vector_search_query)
    ranil_documents = ranil_retriever.get_relevant_documents(ranil_vector_search_query)
    sajith_documents = sajith_retriever.get_relevant_documents(sajith_vector_search_query)

    return {"namal_vector_search_documents": namal_documents,
             "ranil_vector_search_documents": ranil_documents, 
             "sajith_vector_search_documents": sajith_documents}
