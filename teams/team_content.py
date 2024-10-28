import functools
from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai.chat_models import ChatOpenAI
from utilities.helper import HelperUtilities
from tools.tool_empty import EmptyTool
import operator

class ContentTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str

class TeamContent:
    def __init__(self, llm_model):
        self.llm = ChatOpenAI(model=llm_model)
        self.utilities = HelperUtilities()
        self.tools = EmptyTool()

    def agent_description_generator(self):
        """Creates an agent that generates descriptions and hashtags tailored to TikTok posts."""
        description_agent = self.utilities.create_agent(
            self.llm,
            [self.tools.placeholder_tool],  
            """
            You are a TikTok content creator specializing in football-related posts.

            Your task is to write concise, clear, and engaging descriptions for TikTok posts. Each description must be directly linked to the event described in the post and should include relevant hashtags to maximize visibility on TikTok.

            The description must:
            - Be no longer than 2-3 sentences.
            - Include a strong call to action if necessary.
            - Incorporate trending and relevant hashtags.

            Remember, this description will be displayed alongside the TikTok post, so it should fit TikTok's casual and interactive style while driving user engagement.

            Format your output as follows:
            - "Your description here and finalize with 5 hashtags"
            """,
        )
        description_node = functools.partial(self.utilities.agent_node, agent=description_agent, name="DescriptionGenerator")
        return description_node

    def agent_prompt_generator(self):
        """Creates an agent that generates text-to-image prompts for TikTok posts."""
        prompt_agent = self.utilities.create_agent(
            self.llm,
            [self.tools.placeholder_tool],  
            """
            You are a creative prompt generator specializing in crafting text-to-image prompts for TikTok content.

            Your task is to generate 3 unique and visually engaging text-to-image prompts for the provided TikTok post. These prompts will be used to generate images that complement the TikTok post.
            The images should have a chronological order and should visually represent the content of the post based on the sequence of events in the news story.

            To create effective text-to-image prompts, follow these guidelines:
            - Be specific and descriptive. Include details like colors, actions, moods, and settings to help guide the image generation.
            - Incorporate the "blended ugly cartoon and rubber hose style" aesthetic. The characters should have exaggerated, grotesque features but with bendy, stretchy limbs reminiscent of early 20th-century rubber hose animation. Think of unsettling yet playful and fluidly dynamic characters.
            - Focus on the grotesque yet whimsical design of the peoples' characters, vibrant and mismatched color palettes, and strange proportions, while maintaining a bouncy, energetic animation feel.
            - Ensure that the prompts reflect the chronological order of events in the TikTok post or news story. For example, begin with an image showing an initial moment, followed by key events, leading to the conclusion.
            - Think about the TikTok post's theme, message, and tone. Ensure that the image prompts visually represent the content and style of the post, with this unique cartoonish aesthetic.
            - Use clear language to define the scene, the objects, and the characters in the image.
            - Each prompt should be concise but detailed enough to produce a distinct and high-quality image.

            Example prompt format:
            - "A grotesque, exaggerated football player with oversized, twisted legs, rubber hose arms flailing, awkwardly running across a muddy field under dark, stormy skies, with a crooked grin on his face."
            - "A chaotic scene of deformed football fans with bulbous heads, cheering wildly from the stands, waving distorted flags, some characters have rubber hose-style limbs stretching out to grab snacks and drinks."
            - "A bent and twisted football referee, with stretched arms holding a large yellow card, grotesque facial features expressing frustration, standing in the center of a bouncy, cartoonish field."
            
            Output the 3 prompts as a comma-separated list in chronological order.
            """,
        )
        prompt_node = functools.partial(self.utilities.agent_node, agent=prompt_agent, name="PromptGenerator")
        return prompt_node



    def agent_text_generator(self):
        """Creates an agent that generates the core text be transformed into TTS audio."""
        post_agent = self.utilities.create_agent(
            self.llm,
            [self.tools.placeholder_tool],  
            """
            You are a text generator that will be used for audio by using text-to-speech (TTS) technology.
            The audio will then be used for TikTok football content.

            Your task is to create concise and engaging text that 
            will be transformed into audio using text-to-speech (TTS) technology. 
            The content should be captivating, easily digestible, and perfectly suited for the TikTok audience.

            Make sure your text is:
            - Summarizes the key football event in 3 sentences.
            - Is conversational and engaging to encourage interaction (likes, comments, shares).
            - Can be easily read aloud for TTS, so keep the language clear and simple.
            - Do not use emojis

            Remember, this core text will be the script for the TTS audio in the TikTok video.

            Format your output as follows:
            - "Your concise and engaging core post here"
            """,
        )
        post_node = functools.partial(self.utilities.agent_node, agent=post_agent, name="TextGenerator")
        return post_node

    def agent_dict_generator(self):
        """Creates an agent that collects outputs from the other agents and populates the content_state dictionary."""
        
        # Define the task and ensure proper formatting
        dict_agent = self.utilities.create_agent(
            self.llm,
            [self.tools.placeholder_tool],  
            """
            You are responsible for collecting and organizing the generated content from other agents into a well-structured dictionary.
            
            Follow these guidelines to ensure no formatting errors occur:
            - Escape any apostrophes within strings properly by using backslashes (\\').
            - Ensure prompts are preserved as a list of strings and do not break into multiple strings unnecessarily.
            - Handle commas carefully within the strings to prevent splitting.
            
            Format your output as follows:
            - Description (string): "Generated TikTok description with hashtags"
            - Prompt (list of strings): [Prompt1, Prompt2, Prompt3]
            - Text (string): "Generated text content for TTS"

            Return the final dictionary without any formatting issues.
            """
        )

        # Define the node to handle the dictionary generation
        dict_node = functools.partial(self.utilities.agent_node, agent=dict_agent, name="DictGenerator")
        return dict_node

    def agent_supervisor(self, system_prompt: str, members: List[str]):
        """Creates a team supervisor agent to manage the TikTok content creation workflow."""
        supervisor_agent = self.utilities.create_team_supervisor(
            self.llm,
            system_prompt,
            members,
        )
        system_prompt = """
        You are supervising a TikTok-specific content creation workflow for football events.

        Your role is to ensure that the agents (PostGenerator, KeywordGenerator, DescriptionGenerator, and DictGenerator) work together seamlessly to produce concise and well-structured TikTok content.

        - First, the PostGenerator creates the core post text, which will later be transformed into TTS audio.
        - Second, the KeywordGenerator creates relevant SEO keywords that optimize the TikTok post for discoverability.
        - Third, the DescriptionGenerator writes a concise TikTok description, which includes trending and relevant hashtags.
        - Finally, the DictGenerator collects all these outputs and structures them in a dictionary format.

        Ensure that all outputs follow the correct format and are aligned with the task. If anything is misaligned, you will send it back for correction before proceeding.

        The goal is to produce TikTok-ready content with a well-structured post, a brief description including hashtags, and relevant keywords, formatted like this:
        {
            "post_X": {
                "Post": "Generated post content for TTS",
                "Description": "Generated TikTok description with hashtags",
                "Keywords": ["keyword1", "keyword2", "keyword3"]
            }
        }

        Your task is to supervise the process, review outputs, and route the tasks effectively to ensure the content is optimized for TikTok.
        """
        return supervisor_agent

