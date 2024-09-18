from langchain.schema import Document
from RAG.tools.vectore_store_retriever import sajith_retriever, namal_retriever, ranil_retriever, anura_retriever


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
    namal_vector_search_query = state["namal_vector_search_query"]
    ranil_vector_search_query = state["ranil_vector_search_query"]
    sajith_vector_search_query = state["sajith_vector_search_query"]
    anura_vector_search_query = state["anura_vector_search_query"]

    # Retrieval
    if namal_vector_search_query:
        namal_documents = namal_retriever.get_relevant_documents(namal_vector_search_query)
    else:
        namal_documents = []
    if ranil_vector_search_query:
        ranil_documents = ranil_retriever.get_relevant_documents(ranil_vector_search_query)
    else:
        ranil_documents = []
    if sajith_vector_search_query:
        sajith_documents = sajith_retriever.get_relevant_documents(sajith_vector_search_query)
    else:
        sajith_documents = []
    if anura_vector_search_query:
        anura_documents = anura_retriever.get_relevant_documents(anura_vector_search_query)
    else:
        anura_documents = []

    return {"namal_vector_search_documents": namal_documents,
             "ranil_vector_search_documents": ranil_documents, 
             "sajith_vector_search_documents": sajith_documents,
             "anura_vector_search_documents": anura_documents}
