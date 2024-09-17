from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel  , Field
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os

class ContextualizeQuestion(BaseModel):
  """Contextualize the question."""

  contextualized_question: str = Field(
      ...,
      description="The contextualized question.",
  )

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY environment variable not set")
llm = ChatMistralAI(model="mistral-large-latest", api_key=os.getenv("MISTRAL_API_KEY"))

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
structured_llm_router = llm.with_structured_output(ContextualizeQuestion)

contextualizer = contextualize_q_prompt | structured_llm_router
