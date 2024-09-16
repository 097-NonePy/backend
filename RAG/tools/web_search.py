from langchain_community.tools import TavilySearchResults

# Load environment variables
# tavily_api_key = "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"
import os

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = "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"
# Initialize the TavilySearchAPIWrapper with the API key
web_search_tool = TavilySearchResults(
    max_results=5,
)