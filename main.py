import getpass
import os
from dotenv import load_dotenv
from pymongo import MongoClient

# set API keys for OpenAI (the LLM we will use) and Tavily (the search tool we will use)
# Load environment variables from .env file
load_dotenv()

# Retrieve MongoDB credentials from environment variables
username = os.getenv("MONGODB_USERNAME")
password = os.getenv("MONGODB_PASSWORD")
cluster = os.getenv("MONGODB_CLUSTER")
database_name = os.getenv("MONGODB_DATABASE")

# Construct MongoDB connection string
connection_string = f"mongodb+srv://{username}:{password}@{cluster}/{database_name}?retryWrites=true&w=majority"

# MongoDB setup
client = MongoClient(connection_string)
db = client[database_name]
news_collection = db['articles']

# Access the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")
hf_api_key = os.getenv("HF_API_KEY")

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
from scraper.scraper import FootballNewsScraper

from utilities.util_main import transform_to_list
from run_research_output import test_output, content_output_test
from utilities.util_texttoimg import generate_images_from_prompts
from utilities.util_tts import text_to_speech_conversion
from utilities.util_video import create_videos_from_posts, postprocess_videos

def run_scraper(api_key, news_collection):
    """football_teams = ["Arsenal", "Manchester United", "Manchester City", "Liverpool",
                      "Chelsea", "Tottenham Hotspur", "Real Madrid", "Barcelona",
                      "Paris Saint-Germain", "Bayern Munich"]"""
    football_teams = ["Arsenal", "Manchester United"] # test
    scraper_system = FootballNewsScraper(api_key, news_collection)
    urls = scraper_system.run(football_teams)
    return urls


# Function to run the research process
def run_research_chain(query):
    # Create an instance of the ResearchChain class
    research_chain = ResearchChain()

    # Build the research graph
    system_prompt = research_sys
    members = ["AgentScrape", "AgentList"]
    research_chain.build_graph(system_prompt, members)

    # Compile the research chain
    compiled_chain = research_chain.compile_chain()

    # Execute the research chain with the provided message
    research_result = research_chain.enter_chain(query, compiled_chain)
    
    # Return the result to be used in the content process
    return research_result

# Function to run the content process
def run_content_chain(list_news):
    # Create an instance of the ContentChain class
    content_chain = ContentChain()

    # Build the content generation graph
    system_prompt = content_sys
    members = ["DescriptionGenerator", "PromptGenerator", "TextGenerator"]
    content_chain.build_graph(system_prompt, members)

    # Compile the content creation chain
    compiled_chain = content_chain.compile_chain()

    # Execute the content chain with the list generator content
    final_post_data = content_chain.enter_chain(list_news, compiled_chain)
    
    return final_post_data


def main():
    # Step 1: Run the scraper to get the URLs
    news_api_key = os.getenv("NEWS_API_KEY")
    urls = run_scraper(news_api_key, news_collection)
    #print("URLs:", urls)

    """urls = (
        "Opposition View Q&A: De Ligt and Mazraoui transfers"
        "Rayyan from Bavarian Football Works answers questions about United’s two newest signings"
        "https://thebusbybabe.sbnation.com/2024/8/17/24222506/opposition-view-q-a-de-ligt-and-mazraoui-transfers"
        ""
        "Barcelona boss Flick hopeful Gündogan stays after talks"
        "Barcelona boss Hansi Flick is hopeful Ilkay Gündogan will stay at the club after holding talks with the midfielder."
        "https://www.espn.com/soccer/story/_/id/40898109/barcelona-hansi-flick-ilkay-gundogan-talks-stay-transfer-window"
    )"""

    # Step 2: Run the research chain using the determined query
    output = run_research_chain(urls)
    #print("List Generator Content:", output)
    #output = test_output
    #output_list = transform_to_list(output)

    # Step 3: Run the content chain using the result from the research chain
    output_content = run_content_chain(test_output)
    # Print the final post data (containing post, description, and hashtags)
    print("Final post data:", output_content)

    # Step 4: Turn text into audio using TTS technology
    text_to_speech_conversion(content_output_test, key="Text", folder_name="audio")
    
    # Step 5: Generate images from the prompts
    generate_images_from_prompts(content_output_test, folder_name="images", api_key=hf_api_key)
    
    # Step 6: Create videos from the posts
    # Example of running the function for post_1
    create_videos_from_posts(content_output_test)

    # Example usage: Post-process the videos generated earlier
    postprocess_videos(input_folder='output', output_folder='processed_videos')



if __name__ == "__main__":
    main()