import operator
from pathlib import Path
from typing import Annotated, List, TypedDict
from langchain_openai.chat_models import ChatOpenAI
from utilities.helper import HelperUtilities
from tools.tool_writer import WriterTool
from langchain_core.messages import BaseMessage
import functools

class TeamWriter:
    class DocWritingState(TypedDict):
        # This tracks the team's conversation internally
        messages: Annotated[List[BaseMessage], operator.add]
        # This provides each worker with context on the others' skill sets
        team_members: str
        # This is how the supervisor tells langgraph who to work next
        next: str
        # This tracks the shared directory state
        current_files: str

    def __init__(self, llm_model):
        self.llm = ChatOpenAI(model=llm_model)
        self.utilities = HelperUtilities()
        self.writer_tool = WriterTool()  # Correct tool class
        self.working_directory = Path.cwd()  # Use the current working directory

    def prelude(self, state: DocWritingState):
        """This will be run before each worker agent begins work.
        It updates the working directory state for the agents."""
        written_files = []
        if not self.working_directory.exists():
            self.working_directory.mkdir()
        try:
            written_files = [
                f.relative_to(self.working_directory) for f in self.working_directory.rglob("*")
            ]
        except Exception:
            pass

        if not written_files:
            return {**state, "current_files": "No files written."}

        return {
            **state,
            "current_files": "\nBelow are files your team has written to the directory:\n"
            + "\n".join([f" - {f}" for f in written_files]),
        }

    def agent_writer(self):
        """Creates the writing agent responsible for writing and editing documents."""
        content_writer = self.utilities.create_agent(
            self.llm,
            [self.writer_tool.write_document, self.writer_tool.edit_document, self.writer_tool.read_document],
            "You are an expert writer tasked with writing and editing documents.\n{current_files}",
        )
        return functools.partial(self.utilities.agent_node, agent=content_writer, name="DocWriter")

    def agent_note_taker(self):
        """Creates a note-taking agent for outlining and notes."""
        note_taker = self.utilities.create_agent(
            self.llm,
            [self.writer_tool.create_outline, self.writer_tool.read_document],
            "You are an expert tasked with creating outlines and taking notes for a research paper.\n{current_files}",
        )
        return functools.partial(self.utilities.agent_node, agent=note_taker, name="NoteTaker")

    def agent_chart_generator(self):
        """Creates the chart-generating agent."""
        chart_generator = self.utilities.create_agent(
            self.llm,
            [self.writer_tool.read_document, self.writer_tool.python_repl],
            "You are a data visualization expert tasked with generating charts for the research project.\n{current_files}",
        )
        return functools.partial(self.utilities.agent_node, agent=chart_generator, name="ChartGenerator")

    def supervisor(self, members: List[str]):
        """Creates the supervisor agent to manage the team's workflow."""
        return self.utilities.create_team_supervisor(
            self.llm,
            "You are the supervisor managing the following workers: {team_members}. Assign the next task based on the current state.",
            members,
        )

    def create_team(self):
        """Creates the full team of agents and connects them to the supervisor."""
        members = ["DocWriter", "NoteTaker", "ChartGenerator"]

        # Create the agents and supervisor
        doc_writer_agent = self.agent_writer()
        note_taker_agent = self.agent_note_taker()
        chart_generator_agent = self.agent_chart_generator()
        doc_writing_supervisor = self.supervisor(members)

        # Example usage of connecting the agents to the workflow
        return doc_writer_agent, note_taker_agent, chart_generator_agent, doc_writing_supervisor
