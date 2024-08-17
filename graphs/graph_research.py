from langgraph.graph import StateGraph, START, END
from teams.team_research import TeamResearch
from teams.team_writer import TeamWriter 
from langchain_core.messages import HumanMessage


class ResearchChain:
    def __init__(self):
        # Create an instance of WorkerAgents
        self.agents = TeamResearch(llm_model="gpt-4-1106-preview")
        self.research_graph = StateGraph(self.agents.ResearchTeamState)  # Initialize the StateGraph

    def build_graph(self, 
                    system_prompt: str = "Supervising the workflow", 
                    members: list = ["Search", "WebScraper"]):
        """Build the research graph by adding nodes and edges."""
        # Add nodes using agent methods
        self.research_graph.add_node("Search", self.agents.agent_search())
        self.research_graph.add_node("WebScraper", self.agents.agent_research())
        self.research_graph.add_node("supervisor", self.agents.agent_supervisor(system_prompt, members))

        # Add edges between nodes
        self.research_graph.add_edge("Search", "supervisor")
        self.research_graph.add_edge("WebScraper", "supervisor")

        # Add conditional edges for dynamic routing
        self.research_graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {"Search": "Search", "WebScraper": "WebScraper", "FINISH": END},
        )

        # Set the starting point of the graph
        self.research_graph.add_edge(START, "supervisor")

    def compile_chain(self):
        """Compile the research chain from the constructed graph."""
        return self.research_graph.compile()

    def enter_chain(self, message: str, chain):
        """Enter the compiled chain with the given message."""
        # Create a list of HumanMessage instances
        results = [HumanMessage(content=message)]

        # Execute the chain by passing the messages to it
        research_chain = chain.invoke({"messages": results})

        return research_chain

