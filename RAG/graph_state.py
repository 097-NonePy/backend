from typing import List

from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    vector_search_query: str
    web_search_query: str
    generation: str

    vector_search_documents: List[str]
    web_search_documents: List[str]

    generated_count: int