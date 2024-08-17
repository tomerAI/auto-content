import functools
from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai.chat_models import ChatOpenAI
from utilities.helper import HelperUtilities
from tools.tool_empty import EmptyTool
import operator

class TeamContent:
    class ContentTeamState(TypedDict):
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
        self.utilities = HelperUtilities()  # Helper utilities instance
        self.tools = EmptyTool()  # Empty tool instance
        
    def agent_description_generator(self):
        """Creates an agent that generates TikTok descriptions from raw content."""
        description_agent = self.utilities.create_agent(
            self.llm,
            [self.tools.placeholder_tool],  
            "You are a content creator who specializes in writing engaging TikTok descriptions from raw content.",
        )
        description_node = functools.partial(self.utilities.agent_node, agent=description_agent, name="DescriptionGenerator")
        return description_node

    def agent_hashtags_generator(self):
        """Creates an agent that generates hashtags from the raw content."""
        hashtags_agent = self.utilities.create_agent(
            self.llm,
            [self.tools.placeholder_tool],  
            "You are a social media expert who generates relevant and trending hashtags for TikTok posts.",
        )
        hashtags_node = functools.partial(self.utilities.agent_node, agent=hashtags_agent, name="HashtagsGenerator")
        return hashtags_node

    def agent_post_generator(self):
        """Creates an agent that generates TikTok posts from researched material."""
        metadata_agent = self.utilities.create_agent(
            self.llm,
            [self.tools.placeholder_tool],  
            "You are a content strategist who creates metadata, including tags, categories, and keywords, for TikTok posts.",
        )
        metadata_node = functools.partial(self.utilities.agent_node, agent=metadata_agent, name="MetadataGenerator")
        return metadata_node

    def agent_supervisor(self, system_prompt: str, members: List[str]):
        """Creates a team supervisor agent to manage the conversation between workers."""
        supervisor_agent = self.utilities.create_team_supervisor(
            self.llm,
            system_prompt,
            members,
        )
        return supervisor_agent
