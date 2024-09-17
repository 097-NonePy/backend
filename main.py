import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")


from fastapi import FastAPI, Request
from RAG.graph import app as rag_app
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

config = {"configurable": {"thread_id": "1"}}
app = FastAPI()


load_dotenv()
memory = ConversationBufferMemory()
# Create a dictionary to store ConversationBufferMemory instances for each thread
thread_memories = {}

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def test():
    from pprint import pprint

    # Run
    inputs = {
        "question": "What is the differnce between sajith premadasa's actions for the health sector and ranil wickramasinghe's actions for the health sector?"
    }

    for output in rag_app.stream(inputs, config=config):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            pprint(value, indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    # pprint(value["generation"])
    # return value["generation"]
    return "Hello World"

@app.post("/compare")
async def compare(request: dict):
    from pprint import pprint

    instructions = request.get("instructions")
    field = request.get("field")

    compare_2019 = request.get("compare_2019")

    if not instructions and not field:
        return {"error": "Either instructions or field must be provided"}

    if not instructions and field == 'misc':
        return {"error": "Miscellaneous field requires instructions"}

    candidates = []
    if request.get("namal") is True:
        candidates.append("namal rajapakse")
    if request.get("sajith") is True:
        candidates.append("sajith premadasa")
    if request.get("ranil") is True:
        candidates.append("ranil wickramasinghe")

    if len(candidates) == 0:
        return {"error": "At least one candidate must be provided"}

    field_instructions = f"What are the key differences between candidates on approaching the {field}?"

    question = f"""You need to focus on the following candidates: {' '.join(candidates)}. 
    You need to answer the question based on the provided instructions and the field.
    {"Also compare with the 2019 election manifesto of the candidates" if compare_2019 else ""}
    {instructions if instructions else field_instructions}"""

    print(question)

    inputs = {
        "question": question
    }

    for output in rag_app.stream(inputs):
        for key, value in output.items():
            pprint(f"Node '{key}':")
            pprint(value, indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    pprint(value["generation"])
    return {"answer": value["generation"]}
    
@app.post("/process")
async def process(request: Request):
    from pprint import pprint

    data = await request.json()
    question = data.get("question", "")
    # Run
    inputs = {
        "question": question,
        "chat_history": memory.load_memory_variables({})["history"]
    }

    response = []
    for output in rag_app.stream(inputs, config=config):
        node_output = {}
        for key, value in output.items():
            # Node
            node_output[key] = value
        response.append(node_output)

    return response

@app.post("/chat")
async def chat(request: dict):
    from pprint import pprint
    question = request.get("question")
    thread_id = request.get("thread_id")

     # Get or create a ConversationBufferMemory instance for this thread
    if thread_id not in thread_memories:
        thread_memories[thread_id] = ConversationBufferMemory(return_messages=True)
    
    memory = thread_memories[thread_id]
    chat_history = memory.load_memory_variables({})["history"]

    # Prepare inputs with chat history
    inputs = {
        "question": question,
        "chat_history": chat_history
    }

    config = {"configurable": {"thread_id": thread_id}}
    for output in rag_app.stream(inputs, config=config):
        for key, value in output.items():
            pprint(f"Node '{key}':")
            pprint(value, indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    # pprint(value["generation"])
    # return {"answer": value["generation"]}
    if final_output and "generation" in final_output:
        # Add the interaction to memory
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(final_output["generation"])
    return value

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)