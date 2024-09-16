from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

file_names = ["namal.txt", "ranil.txt", "sajith.txt"]
file_name_to_source = {
    "namal.txt": "namal",
    "ranil.txt": "ranil",
    "sajith.txt": "sajith",
}
docs = []

for file_name in file_names:
    loader = TextLoader(f"documents/{file_name}", encoding="utf-8") 
    doc = loader.load()
    for d in doc:
        d.metadata = {"file_name": file_name, "source": file_name_to_source[file_name]}
        docs.append(d)

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)

# Save the FAISS index to disk
vectorstore.save_local("./vectore_stores/manifesto_vectorstore")

sajith_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6, "filter": {"source": "sajith"}})
namal_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6, "filter": {"source": "namal"}})
ranil_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6, "filter": {"source": "ranil"}})