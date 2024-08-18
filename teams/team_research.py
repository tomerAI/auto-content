import functools
import operator
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage
from langchain_openai.chat_models import ChatOpenAI
from utilities.helper import HelperUtilities
from tools.tool_scraper import ToolSearch, ToolResearch
from tools.tool_empty import EmptyTool

class ResearchTeamState(TypedDict):
    # A message is added after each team member finishes
    messages: Annotated[List[BaseMessage], operator.add]
    # The team members are tracked so they are aware of
    # the others' skill sets
    team_members: List[str]
    # Used to route work. The supervisor calls a function
    # that will update this every time it makes a decision
    next: str


class TeamResearch:
    def __init__(self, llm_model):
        self.llm = ChatOpenAI(model=llm_model)
        self.utilities = HelperUtilities()  # Create an instance of HelperUtilities
        self.empty_tool = EmptyTool()
        self.search_tool = ToolSearch()  # Instantiate the ScraperStats class

    def agent_search(self):
        """Creates a search agent for conducting research using the Tavily search tool."""
        # Improved system prompt with more context and instructions
        system_prompt = (
            "You are a highly capable research assistant specializing in retrieving up-to-date information. "
            "Your task is to conduct thorough searches using the Tavily search engine. "
            "Focus on obtaining detailed and credible information related to football statistics, league data, "
            "and player performances in the Danish Superliga."
        )

        search_agent = self.utilities.create_agent(
            self.llm,
            [self.search_tool.tavily_tool],
            system_prompt,
        )
        search_node = functools.partial(self.utilities.agent_node, agent=search_agent, name="Search")
        return search_node

    def agent_research(self):
        """Creates a research agent for scraping web pages for more detailed information."""
        # Improved system prompt for the research agent
        system_prompt = (
            "You are a research assistant specialized in extracting detailed information from web pages. "
            "Scrape the specified URLs to gather comprehensive football statistics, player data, and match details "
            "from the Danish Superliga. Ensure accuracy and summarize the most relevant points."
        )

        research_agent = self.utilities.create_agent(
            self.llm,
            [ToolResearch],
            system_prompt,
        )
        research_node = functools.partial(self.utilities.agent_node, agent=research_agent, name="WebScraper")
        return research_node

    def agent_list_generator(self):
        """Creates an agent that aggregates news and compiles it into a list."""
        system_prompt = (
            "You are a list generator that aggregates all the news gathered by the team. "
            "Your task is to create a well-structured list of football-related news items based on the data "
            "provided by the Search and WebScraper agents. Ensure that each item is concise and categorized properly."
        )

        list_agent = self.utilities.create_agent(
            self.llm,
            [self.empty_tool.placeholder_tool],  # Placeholder as no tool is needed for aggregation
            system_prompt,
        )
        list_node = functools.partial(self.utilities.agent_node, agent=list_agent, name="ListGenerator")
        return list_node
        
    def agent_supervisor(self, system_prompt: str, members: List[str]):
        """Creates a team supervisor agent to manage the conversation between workers."""
        # Improved system prompt for the supervisor
        system_prompt = (
            "You are the team supervisor responsible for coordinating the tasks between the team members. "
            "Ensure that the Search and WebScraper agents collaborate effectively. Once their tasks are complete, "
            "trigger the ListGenerator agent to compile the final list of football news."
        )

        supervisor_agent = self.utilities.create_team_supervisor(
            self.llm,
            system_prompt,
            members,
        )
        return supervisor_agent