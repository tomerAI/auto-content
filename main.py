import getpass
import os

# set API keys for OpenAI (the LLM we will use) and Tavily (the search tool we will use)
# Access the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

# Set the API keys 
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ['USER_AGENT'] = 'myagent'

# set API key for LangSmith tracing, which will give us best-in-class observability.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Hierarchical Football Content Team"

# Import chains
from graphs.graph_research import ResearchChain
from graphs.graph_content import ContentChain  # Updated to use ContentChain

def main():
    
    # Create an instance of the ResearchChain class
    research_chain = ResearchChain()

    # Build the research chain graph
    research_chain.build_graph()

    # Compile the research chain
    research_chain_compiled = research_chain.compile_chain()

    # Example message that would be passed into the research chain for processing
    research_message = "Please generate content about the latest football statistics in the Danish Superliga."

    # Execute the research chain with the provided message
    research_result = research_chain.enter_chain(research_message, research_chain_compiled)

    # Output the result of the research chain's execution
    print("Research chain result:", research_result)

    # Extract the research result's content
    research_content = research_result['messages'][1:].content

    # Create an instance of the ContentChain class
    content_chain = ContentChain()

    # Build the content chain graph
    content_chain.build_graph()

    # Compile the content chain
    content_chain_compiled = content_chain.compile_chain()

    # Pass the research result to the content chain
    content_message = f"Create a TikTok description, hashtags, and metadata for the following content: {research_content}"

    # Execute the content chain with the provided message
    content_result = content_chain.enter_chain(content_message, content_chain_compiled)

    # Output the result of the content chain's execution
    print("Content chain result:", content_result)

if __name__ == "__main__":
    main()

