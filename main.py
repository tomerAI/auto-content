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

# Import the classes and functions from the graphs and prompts modules
from graphs.graph_research import ResearchChain
from graphs.graph_content import ContentChain
from prompts.prompt_content import content_sys
from prompts.prompt_research import research_sys

from datetime import datetime, timedelta

# Function to determine the initial query
def determine_initial_query():
    # Logic to determine the query
    # This could be based on user input, another function, or a preset query

    # Create variable for yesterday's date
    yesterday = datetime.now() - timedelta(days=1)
    # Format the date to 'yyyy-mm-dd'
    yesterday_str = yesterday.strftime('%Y-%m-%d')

    query = f"Fodboldnyheder fra {yesterday_str}"
    return query, yesterday_str

# Function to run the research process
def run_research_chain(query):
    # Create an instance of the ResearchChain class
    research_chain = ResearchChain()

    # Build the research graph
    system_prompt = research_sys
    members = ["Search", "WebScraper", "ListGenerator"]
    research_chain.build_graph(system_prompt, members)

    # Compile the research chain
    compiled_chain = research_chain.compile_chain()

    # Execute the research chain with the provided message
    research_result = research_chain.enter_chain(query, compiled_chain)

    print("Extracted ListGenerator Content:", research_result)
    
    # Return the result to be used in the content process
    return research_result

# Function to run the content process
def run_content_chain(list_news):
    # Create an instance of the ContentChain class
    content_chain = ContentChain()

    # Build the content generation graph
    system_prompt = content_sys
    members = ["DescriptionGenerator", "KeywordGenerator", "PostGenerator"]
    content_chain.build_graph(system_prompt, members)

    # Compile the content creation chain
    compiled_chain = content_chain.compile_chain()

    # Execute the content chain with the list generator content
    final_post_data = content_chain.enter_chain(list_news, compiled_chain)

    # Print the final post data (containing post, description, and hashtags)
    print("Final post data:", final_post_data)
    
    return final_post_data

def main():
    # Step 1: Determine the initial query
    query = determine_initial_query()

    # Step 2: Run the research chain using the determined query
    list_generator_content = run_research_chain(query)

    # Step 3: Run the content chain using the result from the research chain
    run_content_chain(list_generator_content)

if __name__ == "__main__":
    main()

