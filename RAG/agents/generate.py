from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

template = """
{chat_history}

You are a very vigilant and helpful journalism assistant. Use the following pieces of 
context to answer the question at the end. Your focus is 2024 presidential election in Sri Lanka.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Be concise and helpful. Limit your response to 2-3 sentences. Like a normal text conversation.

________________________________________________________________________________
Here are the web results regarding the question:
{web_context}

Here are the results from the manifesto of the candidates:
________________________________________________________________________________
namel rajapakse:
{namal_context}

________________________________________________________________________________
ranil wickramasinghe:
{ranil_context}

________________________________________________________________________________
sajith premadasa:
{sajith_context}

________________________________________________________________________________
Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

# LLM
llm = ChatOpenAI(model="gpt-4o-mini")


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = custom_rag_prompt | llm | StrOutputParser()