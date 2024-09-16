from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel  , Field
from langchain_mistralai import ChatMistralAI
import os
template = """You are a very vigilant and helpful journalist. Use the following pieces of 
context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Be concise and helpful. 

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
llm = ChatMistralAI(model="mistral-large-latest", api_key=os.getenv("MISTRAL_API_KEY"))

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = custom_rag_prompt | llm | StrOutputParser()