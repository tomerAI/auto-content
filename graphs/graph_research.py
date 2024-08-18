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

        def list_generator_callback(state):
            messages = state["messages"]

            # Extract content from all previous messages (Search, WebScraper, etc.)
            aggregated_content = []
            for msg in messages:
                if msg.name in ["Search", "WebScraper"]:
                    aggregated_content.append(msg.content)
            
            # Concatenate the aggregated content into a single news item
            news_item = "\n".join(aggregated_content)

            # Add the aggregated news item to the news list
            self.news_list.append(news_item)

            return state

        # Add nodes using agent methods
        self.research_graph.add_node("Search", self.agents.agent_search())
        self.research_graph.add_node("WebScraper", self.agents.agent_research())
        self.research_graph.add_node(
            "ListGenerator", 
            functools.partial(self.agents.agent_list_generator(), callback=list_generator_callback)
        )
        self.research_graph.add_node("supervisor", self.agents.agent_supervisor(system_prompt, members))

        # Add edges between nodes
        self.research_graph.add_edge("Search", "supervisor")
        self.research_graph.add_edge("WebScraper", "supervisor")
        self.research_graph.add_edge("ListGenerator", "supervisor")

        # Add conditional edges for dynamic routing
        self.research_graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {"Search": "Search", "WebScraper": "WebScraper", "ListGenerator": "ListGenerator", "FINISH": END},
        )

        # Set the starting point of the graph
        self.research_graph.add_edge(START, "supervisor")
        self.research_graph.add_edge("ListGenerator", END)

    def compile_chain(self):
        """Compile the research chain from the constructed graph."""
        return self.research_graph.compile()

    def enter_chain(self, message: str, chain):
        """Enter the compiled chain with the given message."""
        # Create a list of HumanMessage instances
        results = [HumanMessage(content=message)]

        # Execute the chain by passing the messages to it
        research_chain = chain.invoke({"messages": results})

        # Return the final populated news list
        return self.news_list
