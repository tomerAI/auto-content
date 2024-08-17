from langgraph.graph import StateGraph, START, END
from teams.team_content import TeamContent  # Assuming this is the new team you created
from langchain_core.messages import HumanMessage

class ContentChain:
    def __init__(self):
        # Create an instance of ContentCreationAgents (TeamContentCreation)
        self.agents = TeamContent(llm_model="gpt-4-1106-preview")  # Using GPT-4 for this chain
        self.content_graph = StateGraph(self.agents.ContentTeamState)  # Initialize the StateGraph for content creation

    def build_graph(self,
                    system_prompt: str = "Supervising the content creation workflow",
                    members: list = ["DescriptionGenerator", "HashtagsGenerator", "MetadataGenerator", "SQLSaver"]):
        """Build the content graph by adding nodes and edges."""
        # Add nodes using agent methods for content creation
        self.content_graph.add_node("DescriptionGenerator", self.agents.agent_description_generator())
        self.content_graph.add_node("HashtagsGenerator", self.agents.agent_hashtags_generator())
        self.content_graph.add_node("PostGenerator", self.agents.agent_post_generator())
        self.content_graph.add_node("supervisor", self.agents.agent_supervisor(system_prompt, members))

        # Add edges between nodes to define workflow
        self.content_graph.add_edge("DescriptionGenerator", "supervisor")
        self.content_graph.add_edge("HashtagsGenerator", "supervisor")
        self.content_graph.add_edge("PostGenerator", "supervisor")

        # Add conditional edges for dynamic routing based on the supervisor's decision
        self.content_graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {
                "DescriptionGenerator": "DescriptionGenerator",
                "HashtagsGenerator": "HashtagsGenerator",
                "PostGenerator": "PostGenerator",
                "FINISH": END,
            },
        )

        # Set the starting point of the graph
        self.content_graph.add_edge(START, "supervisor")

    def compile_chain(self):
        """Compile the content creation chain from the constructed graph."""
        return self.content_graph.compile()

    def enter_chain(self, message: str, chain):
        """Enter the compiled chain with the given message."""
        # Create a list of HumanMessage instances
        results = [HumanMessage(content=message)]

        # Execute the chain by passing the messages to it
        content_chain = chain.invoke({"messages": results})

        return content_chain
