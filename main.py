import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

from fastapi import FastAPI, Request
from RAG.graph import app as rag_app

from dotenv import load_dotenv
import os


app = FastAPI()


load_dotenv()

@app.get("/")
async def test():
    from pprint import pprint

    # Run
    inputs = {
        "question": "What is the differnce between sajith premadasa's actions for the health sector and ranil wickramasinghe's actions for the health sector?"
    }

    for output in rag_app.stream(inputs):
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

@app.post("/process")
async def process(request: Request):
    from pprint import pprint

    data = await request.json()
    question = data.get("question", "")

    # Run
    inputs = {
        "question": question
    }

    response = []
    for output in rag_app.stream(inputs):
        node_output = {}
        for key, value in output.items():
            # Node
            node_output[key] = value
        response.append(node_output)

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)