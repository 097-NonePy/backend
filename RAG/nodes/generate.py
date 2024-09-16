from RAG.agents.generate import rag_chain

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    web_documents = state["web_search_documents"]
    vector_documents = state["vector_search_documents"]

    documents = web_documents + vector_documents

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})

    generated_count = state.get("generated_count", 0) + 1
    return {
        "web_search_documents": web_documents, 
        "vector_search_documents": vector_documents, 
        "question": question, 
        "generation": generation,
        "generated_count": generated_count
        }