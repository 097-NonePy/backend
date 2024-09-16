from langchain_community.document_loaders import TextLoader
import os

file_names = ["namal.txt", "ranil.txt", "sajith.txt"]
file_name_to_source = {
    "namal.txt": "namal",
    "ranil.txt": "ranil",
    "sajith.txt": "sajith",
}

docs = []

for file_name in file_names:
    loader = TextLoader(f"documents/{file_name}") 
    doc = loader.load()
    for d in doc:
        d.metadata = {"file_name": file_name}
        docs.append(d)

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents=all_splits, embedding=OpenAIEmbeddings(), persist_directory="./vectore_stores/manifesto_vectorstore",
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})