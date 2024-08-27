from langgraph.graph import StateGraph, START, END
from teams.team_research import TeamResearch, ResearchTeamState
from teams.team_writer import TeamWriter
from langchain_core.messages import HumanMessage
import functools

class ResearchChain:
    def __init__(self):
        # Create an instance of WorkerAgents
        self.agents = TeamResearch(llm_model="gpt-4-1106-preview")
        self.research_graph = StateGraph(ResearchTeamState)  # Initialize the StateGraph

        self.news_list = []  # Use a list to store aggregated news items

    def build_graph(self, system_prompt, members):
        """Build the research graph by adding nodes and edges."""

        # Add nodes using agent methods
        self.research_graph.add_node("AgentScrape", self.agents.agent_research())
        self.research_graph.add_node("AgentList", self.agents.agent_list())
        self.research_graph.add_node("supervisor", self.agents.agent_supervisor(system_prompt, members))

        # Add conditional edges for dynamic routing
        self.research_graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {"AgentScrape": "AgentScrape", 
             "AgentList": "AgentList", 
             "FINISH": END},
        )

        # Manual graph construction
        self.research_graph.add_edge(START, "supervisor")
        self.research_graph.add_edge("supervisor", "AgentScrape")
        self.research_graph.add_edge("AgentScrape", "AgentList")
        self.research_graph.add_edge("AgentList", END)

    def compile_chain(self):
        """Compile the research chain from the constructed graph."""
        return self.research_graph.compile()

    def enter_chain(self, message: str, chain):
        """Enter the compiled chain with the given message and return the last message from the chain."""
        # Create a list of HumanMessage instances
        results = [HumanMessage(content=message)]

        # Execute the chain by passing the messages to it
        research_chain = chain.invoke({"messages": results})

        # Assuming the chain returns a list of messages, get the last message
        if "messages" in research_chain and research_chain["messages"]:
            last_message = research_chain["messages"][-1].content
        else:
            last_message = "No valid messages returned from the chain."

        # Return the last message as the final output
        return last_message

