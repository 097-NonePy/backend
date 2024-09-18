from typing import List

from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages.base import BaseMessage


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    namal_vector_search_query: str
    ranil_vector_search_query: str
    sajith_vector_search_query: str
    anura_vector_search_query: str
    web_search_query: str
    generation: str

    namal_vector_search_documents: List[str]
    ranil_vector_search_documents: List[str]
    sajith_vector_search_documents: List[str]
    anura_vector_search_documents: List[str]
    web_search_documents: List[str]

    generated_count: int

    chat_history: Annotated[List[BaseMessage], add_messages]
    contextualized_question: str
