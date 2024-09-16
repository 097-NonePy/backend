### Router

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


# Data model
class ExtractQuery(BaseModel):
    """Route a user query to the relevant datasources with subquestions."""

    vector_search_query: str = Field(
        ...,
        description="The query to search the vector store.",
    )
    web_search_query: str = Field(
        ...,
        description="The query to search the web.",
    )

llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm_router = llm.with_structured_output(ExtractQuery)

# Prompt
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to Manifests of political candidate Sajith Premadasa.
for an example, what this candidate do for education sector, health sector etc is on the vectorstore.
And also their plans for the future of the country is on the vectorstore.

If the question involves something about sajith's policies in a past year, then you will have to search.
And also if you feel like a web search will be usefull. Do a web search.

After deciding,
Output the 'vector_search_query': The query that needs to be searched from the vector store.
And the 'web_search_query': The query that needs to be searched from the web.
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_extractor = route_prompt | structured_llm_router