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
    """
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
    research_content = research_result['messages'][1:].content"""

    news_items = [
        "Kasper Hjulmand resigns as Denmark's manager after Euro 2024 exit.",
        "F.C. København wins their 15th Danish Superliga title.",
        "Christian Eriksen returns to Denmark's squad for World Cup qualifiers after recovering from his cardiac arrest."
    ]

    # Create an instance of the ContentChain class
    content_chain = ContentChain()

    # Step 2: Build the graph
    system_prompt = """
    You are supervising a content creation workflow. Your role is to oversee and guide the coordination of three specialized agents, ensuring that the tasks are routed efficiently based on their expertise.

    1. **DescriptionGenerator**: This agent is responsible for crafting engaging, concise descriptions that are tailored to TikTok. It uses SEO metadata to enhance visibility and maximize audience engagement, focusing on retention and interaction.

    2. **KeywordGenerator**: This agent is an SEO expert. Its job is to generate relevant keywords and hashtags based on the content and trends. It optimizes posts for discoverability, ensuring the content reaches the right audience.

    3. **PostGenerator**: This agent creates TikTok posts based on provided research material. It focuses on generating engaging, trend-aligned posts that captivate the audience and encourage engagement. The posts are optimized for social media platforms, particularly TikTok.

    Your task is to route work between these agents based on the content creation flow, ensuring that each agent is tasked with their respective specialty at the appropriate time. Once their work is completed, you will determine the next step or finalize the workflow. 
    """
    members = ["DescriptionGenerator", "KeywordGenerator", "PostGenerator"]
    content_chain.build_graph(system_prompt, members)

    # Step 3: Compile the content creation chain
    compiled_chain = content_chain.compile_chain()

    # Step 5: Enter the chain with the initial message and execute it
    final_post_data = content_chain.enter_chain(news_items, compiled_chain)

    # Step 6: Print the final post data (containing post, description, and hashtags)
    print("Final post data:", final_post_data)


if __name__ == "__main__":
    main()

