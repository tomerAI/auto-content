import functools
import operator
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage
from langchain_openai.chat_models import ChatOpenAI
from utilities.helper import HelperUtilities
from tools.tool_scraper import ScraperTool


class TeamResearch:
    class ResearchTeamState(TypedDict):
        # A message is added after each team member finishes
        messages: Annotated[List[BaseMessage], operator.add]
        # The team members are tracked so they are aware of
        # the others' skill sets
        team_members: List[str]
        # Used to route work. The supervisor calls a function
        # that will update this every time it makes a decision
        next: str

    def __init__(self, llm_model):
        self.llm = ChatOpenAI(model=llm_model)
        self.utilities = HelperUtilities()  # Create an instance of HelperUtilities
        self.scraper_tool = ScraperTool()  # Instantiate the ScraperStats class

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
            [self.scraper_tool.tavily_tool],
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
            [self.scraper_tool.scrape_webpages],
            system_prompt,
        )
        research_node = functools.partial(self.utilities.agent_node, agent=research_agent, name="WebScraper")
        return research_node
    
    def agent_supervisor(self, system_prompt: str, members: List[str]):
        """Creates a team supervisor agent to manage the conversation between workers."""
        # Improved system prompt for the supervisor
        system_prompt = (
            "You are the team supervisor responsible for coordinating the tasks between the team members. "
            "You must ensure that the Search and WebScraper agents collaborate effectively. "
            "Guide them to focus on delivering accurate and comprehensive football statistics, and decide "
            "whether to request additional information or finalize the task."
        )

        supervisor_agent = self.utilities.create_team_supervisor(
            self.llm,
            system_prompt,
            members,
        )
        return supervisor_agent
