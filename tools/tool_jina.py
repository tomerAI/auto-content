from langchain_core.tools import tool
import requests
from typing import List

JINA_API_URL = "https://r.jina.ai/"
JINA_API_KEY = "Bearer jina_c4850118b656482fa2f80a639e9d522bBpgNwagrEHMeMNqhu6FXVtmrl0ln"  # Replace with your actual API key

@tool
def ToolResearch(urls: List[str]) -> str:
    """Use Jina's Reader API to extract and format content from the provided web pages."""
    
    # Prepare headers for Jina's Reader API
    headers = {
        "Authorization": JINA_API_KEY
    }
    
    # Step 1: Send each URL to Jina's Reader API and aggregate responses
    formatted_docs = []
    for url in urls:
        api_url = JINA_API_URL + url
        response = requests.get(api_url, headers=headers)
        
        if response.status_code == 200:
            # Directly use the Jina Reader API's formatted response
            formatted_docs.append(response.text)
        else:
            # Handle any errors and include in the formatted output
            formatted_docs.append(f"Error retrieving data from {url}")
    
    # Step 2: Return the aggregated and formatted documents as a single string
    return "\n\n".join(formatted_docs)

