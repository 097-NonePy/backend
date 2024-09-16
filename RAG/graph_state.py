from typing import List, Dict
from langchain_core.messages.base import BaseMessage
from typing_extensions import TypedDict


class Message(TypedDict):
    role: str
    content: str

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    contextualized_question: str
    namal_vector_search_query: str
    ranil_vector_search_query: str
    sajith_vector_search_query: str
    web_search_query: str
    generation: str

    namal_vector_search_documents: List[str]
    ranil_vector_search_documents: List[str]
    sajith_vector_search_documents: List[str]
    web_search_documents: List[str]

    generated_count: int
    chat_history: List[BaseMessage]
