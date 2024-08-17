from langgraph.graph import StateGraph, START, END
from teams.team_writer import TeamWriter
from langchain_core.messages import HumanMessage


class WriterChain:
    def __init__(self):
        # Initialize the writer team
        self.writer_team = TeamWriter(llm_model="gpt-4o")
        
        # Initialize the StateGraph with the DocWritingState
        self.writer_graph = StateGraph(self.writer_team.DocWritingState)  

    def build_graph(self, 
                    system_prompt: str = "Supervising the writing workflow", 
                    writer_members: list = ["DocWriter", "NoteTaker", "ChartGenerator"]):
        """Build the writing graph by adding nodes and edges."""
        
        # Add Writer Team nodes
        self.writer_graph.add_node("DocWriter", self.writer_team.agent_writer())
        self.writer_graph.add_node("NoteTaker", self.writer_team.agent_note_taker())
        self.writer_graph.add_node("ChartGenerator", self.writer_team.agent_chart_generator())
        self.writer_graph.add_node("writer_supervisor", self.writer_team.supervisor(writer_members))

        # Add edges between Writer Team nodes
        self.writer_graph.add_edge("DocWriter", "writer_supervisor")
        self.writer_graph.add_edge("NoteTaker", "writer_supervisor")
        self.writer_graph.add_edge("ChartGenerator", "writer_supervisor")

        # Add conditional edges for dynamic routing
        self.writer_graph.add_conditional_edges(
            "writer_supervisor",
            lambda x: x["next"],
            {
                "DocWriter": "DocWriter",
                "NoteTaker": "NoteTaker",
                "ChartGenerator": "ChartGenerator",
                "FINISH": END
            },
        )

        # Set the starting point of the graph
        self.writer_graph.add_edge(START, "writer_supervisor")

    def compile_chain(self):
        """Compile the writing chain from the constructed graph."""
        return self.writer_graph.compile()

    def enter_chain(self, message: str, chain):
        """Enter the compiled chain with the given message."""
        # Create a list of HumanMessage instances
        results = [HumanMessage(content=message)]

        # Execute the chain by passing the messages to it
        writing_chain = chain.invoke({"messages": results})

        return writing_chain
