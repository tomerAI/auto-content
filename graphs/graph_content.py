from langgraph.graph import StateGraph, START, END
from teams.team_content import TeamContent, ContentTeamState
from langchain_core.messages import HumanMessage
import functools

class ContentChain:
    def __init__(self):
        # Create an instance of ContentCreationAgents (TeamContent)
        self.agents = TeamContent(llm_model="gpt-4-1106-preview")
        self.content_graph = StateGraph(ContentTeamState)

        # Dictionary to store outputs from agents
        self.content_state = {}
        self.post_counter = 1  # Counter to number the posts

    def build_graph(self, system_prompt, members):
        """Build the content graph by adding nodes and edges."""

        def dict_generator_callback(state):
            print("DICT GENERATOR CALLBACK")
            messages = state["messages"]

            # Extract the outputs from the agents
            post_output = next((msg.content for msg in messages if msg.name == "PostGenerator"), "")
            keywords_output = next((msg.content for msg in messages if msg.name == "KeywordGenerator"), "")
            description_output = next((msg.content for msg in messages if msg.name == "DescriptionGenerator"), "")
            
            # Ensure the dictionary entry exists before accessing it
            self.initialize_post_entry()

            post_key = f"post_{self.post_counter}"

            # Add the outputs to the appropriate fields
            self.content_state[post_key]["Post"] = post_output
            self.content_state[post_key]["Keywords"] = keywords_output.split(",")  # Assuming comma-separated keywords
            self.content_state[post_key]["Description"] = description_output

            # Increment the post counter for the next post
            self.post_counter += 1

            print(f"Post {post_key} updated in content_state.")
            print("DICT GENERATOR CALLBACK END")
            return state

        # Create the supervisor agent
        supervisor_agent = self.agents.agent_supervisor(system_prompt, members)

        # Add the agents to the graph
        self.content_graph.add_node("PostGenerator", self.agents.agent_post_generator())
        self.content_graph.add_node("KeywordGenerator", self.agents.agent_keyword_generator())
        self.content_graph.add_node("DescriptionGenerator", self.agents.agent_description_generator())

        # Add the DictGenerator agent with a callback
        self.content_graph.add_node(
            "DictGenerator",
            functools.partial(self.agents.agent_dict_generator(), callback=dict_generator_callback)
        )

        # Add the supervisor agent (without a callback here)
        self.content_graph.add_node("supervisor", supervisor_agent)

        # Set the edges in the correct sequence
        self.content_graph.add_edge(START, "KeywordGenerator")
        self.content_graph.add_edge("KeywordGenerator", "PostGenerator")
        self.content_graph.add_edge("PostGenerator", "DescriptionGenerator")
        self.content_graph.add_edge("DescriptionGenerator", "DictGenerator")
        self.content_graph.add_edge("DictGenerator", "supervisor")
        self.content_graph.add_edge("supervisor", END)

    def compile_chain(self):
        """Compile the content creation chain from the constructed graph."""
        return self.content_graph.compile()

    def initialize_post_entry(self):
        """Initialize the post entry in the content_state."""
        post_key = f"post_{self.post_counter}"
        if post_key not in self.content_state:
            self.content_state[post_key] = {
                "Description": "",
                "Keywords": [],
                "Post": ""
            }

    def enter_chain(self, message: str, chain):
        """Enter the compiled chain with the given message."""
        # Initialize the post entry before starting the chain execution
        self.initialize_post_entry()

        # Create a list of HumanMessage instances
        results = [HumanMessage(content=message)]

        # Execute the chain by passing the messages to it
        content_chain = chain.invoke({"messages": results})

        # Return the final populated content dictionary
        return self.content_state
