from RAG.agents.question_rewriter import question_rewriter

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    
    better_question = question_rewriter.invoke({"question": question})
    return {"question": better_question}