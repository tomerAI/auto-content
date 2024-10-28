import functools
import operator
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage
from langchain_openai.chat_models import ChatOpenAI
from utilities.helper import HelperUtilities
#from tools.tool_scraper import ToolResearch
from tools.tool_jina import ToolResearch
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

    def agent_research(self):
        """Creates a research agent for scraping web pages for more detailed information."""
        # Improved system prompt for the research agent
        system_prompt = (
            "You are a research assistant specialized in extracting detailed information from web pages. "
            "Scrape the specified URLs to gather comprehensive football statistics, player data, and match details "
            "from the best football teams in the world." 
            "Ensure accuracy and summarize the most relevant points."
        )

        research_agent = self.utilities.create_agent(
            self.llm,
            [ToolResearch],
            system_prompt,
        )
        research_node = functools.partial(self.utilities.agent_node, agent=research_agent, name="AgentScrape")
        return research_node

    def agent_list(self):
        """Creates an agent that aggregates news and compiles it into a list of news."""
        system_prompt = (
            "You are a list agent responsible for aggregating and separating all the football-related news gathered "
            "by the research agent. Your task is to create a list where each item is a piece of football news"
            "The news should be comprehensive and engaging making sure to include all the relevant details."
            "Include all the details from AgentScrape and format them into a cohesive output for each news story."
            "Exlucde the source of the news story."
            "The output should be in the form of a Python list wrapped in a string"
            "Here's an example:"
            "['News story 1', 'News story 2', 'News story 3']"
            "Return it as a string to the supervisor agent."
        )

        list_agent = self.utilities.create_agent(
            self.llm,
            [self.empty_tool.placeholder_tool],  # Placeholder as no tool is needed for aggregation
            system_prompt,
        )
        list_node = functools.partial(self.utilities.agent_node, agent=list_agent, name="AgentList")
        return list_node

        
    def agent_supervisor(self, system_prompt: str, members: List[str]):
        """Creates a team supervisor agent to manage the conversation between workers."""
        # Updated system prompt for the supervisor
        system_prompt = (
            "You are the team supervisor responsible for managing the research process. "
            "Your task is to coordinate between the WebScraper and ListGenerator agents. "
            "You will receive URLs, which you should assign to the WebScraper agent to gather information. "
            "Once the WebScraper has successfully completed scraping the content from the provided URLs, "
            "you must then trigger the ListGenerator agent to format the extracted content into a cohesive output. "
            "After the ListGenerator finishes formatting the content, the research task is complete, and you can end the team process."
        )

        supervisor_agent = self.utilities.create_team_supervisor(
            self.llm,
            system_prompt,
            members,
        )
        return supervisor_agent
