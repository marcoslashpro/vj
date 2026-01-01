import json
from uuid import uuid4
import datetime as dt

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime

from typing import Any, Callable
from pathlib import Path

from vj import ASSETS_DIR
from vj.log_utils import get_logger


logger = get_logger()


class LoggingMiddleware(AgentMiddleware):
    def __init__(self, filepath: Path = ASSETS_DIR / "convos.jsonl") -> None:
        super().__init__()

        # File where to save the (sys_p, human_p, answer, available_tools) tuple
        self.filepath = filepath

        # current messages
        self._curr_sys_prompt: SystemMessage | None = None
        self._curr_prompt: HumanMessage | None = None
        self._curr_answer: list[AIMessage | ToolMessage] = []

        # currently available tools
        self._curr_available_tools: list[BaseTool | dict[str, Any]] = []

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        if request.system_prompt:
            logger.debug(f"Extracting system prompt from system_prompt")
            self._curr_sys_prompt = SystemMessage(content=request.system_prompt)
        else:
            logger.debug(f"Extracting system prompt from SystemMessage")
            self._curr_sys_prompt = request.system_message
        logger.debug(f"Extracted system prompt: {self._curr_sys_prompt}")

        logger.debug(f"Extracting available tools: {request.tools}")
        self._curr_available_tools = request.tools

        return handler(request)

    def after_agent(self, state: AgentState, runtime: Runtime[None]) -> None:
        if not len(messages := state["messages"]) > 0:
            logger.debug(f"No messages to process at run end.")
            return

        logger.debug(f"Working with messages: {messages}")
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                logger.debug(f"Extracted prompt message: {msg}")
                self._curr_prompt = msg
                break
            if isinstance(msg, (AIMessage, ToolMessage)):
                self._curr_answer.append(msg)

        self._curr_answer.reverse()
        logger.debug(f"Final extracted answer: {self._curr_answer}")

        self._save_to_file()

    def _save_to_file(self) -> None:
        if not self._curr_prompt:  # Skip saving invalid turn to file
            logger.debug(f"Skipping convo turn with no message prompt.")
            return

        json_obj = {
            "id": uuid4().hex,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "system_prompt": (
                convert_to_openai_messages(self._curr_sys_prompt)
                if self._curr_sys_prompt
                else ""
            ),
            "prompt": convert_to_openai_messages(self._curr_prompt),
            "answer": [convert_to_openai_messages(step) for step in self._curr_answer],
            "available_tools": [
                convert_to_openai_tool(tool) for tool in self._curr_available_tools
            ],
        }
        logger.debug(f"Saving to file: {json_obj}")

        with self.filepath.open(mode="a", encoding="utf-8") as convo_file:
            convo_file.write(json.dumps(json_obj))
