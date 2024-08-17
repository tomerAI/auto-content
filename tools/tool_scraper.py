from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool


class ScraperTool:
    def __init__(self, max_results: int = 5):
        self.tavily_tool = TavilySearchResults(max_results=max_results)

    @tool
    def scrape_webpages(self, urls: List[str]) -> str:
        """Scrape the specified URLs for more detailed information."""
        loader = WebBaseLoader(urls)
        docs = loader.load()
        return "\n\n".join([f'\n{doc.page_content}\n' for doc in docs])

