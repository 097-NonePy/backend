### Router

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel  , Field
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os
# Data model
class ExtractQuery(BaseModel):
    """Route a user query to the relevant datasources with subquestions."""

    namal_vector_search_query: str = Field(
        ...,
        description="The query to search the vector store of namal.",
    )
    ranil_vector_search_query: str = Field(
        ...,
        description="The query to search the vector store of ranil.",
    )
    sajith_vector_search_query: str = Field(
        ...,
        description="The query to search the vector store of sajith.",
    )
    web_search_query: str = Field(
        ...,
        description="The query to search the web.",
    )

load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")

if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY environment variable not set")

# Initialize the ChatMistralAI client with the API key
llm = ChatMistralAI(model="mistral-large-latest", api_key=mistral_api_key)
structured_llm_router = llm.with_structured_output(ExtractQuery)

# Prompt
system = """You are an expert at routing a user question to a vectorstore or web search.
There are three vectorstores. One contains documents related to Manifests of political candidate Sajith Premadasa.
Another contains documents related to Manifests of political candidate Namal Rajapaksa.
The third contains documents related to Manifests of political candidate Ranil Wickramasinghe.

for an example, what this candidate do for education sector, health sector etc is on the vectorstore.
And also their plans for the future of the country is on the vectorstore.

If the question involves something about a candidate's policies in a past year, then you will have to do a websearch.
And also if you feel like a web search will be usefull. Do a web search.

After deciding,
Output the 'namal_vector_search_query': The query that needs to be searched from the vector store of namal.
And the 'ranil_vector_search_query': The query that needs to be searched from the vector store of ranil.
And the 'sajith_vector_search_query': The query that needs to be searched from the vector store of sajith.
And the 'web_search_query': The query that needs to be searched from the web.
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_extractor = route_prompt | structured_llm_router