from fastapi import FastAPI
from RAG.graph import app as rag_app

app = FastAPI()

from dotenv import load_dotenv

load_dotenv()

@app.get("/")
async def test():
    from pprint import pprint

    # Run
    inputs = {
        "question": "What is the differnce between sajith premadasa's actions for the health sector in this election and previous 2019 election?"
    }

    for output in rag_app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            pprint(value, indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    pprint(value["generation"])
    return value