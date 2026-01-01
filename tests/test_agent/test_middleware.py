import json
import tempfile
from pathlib import Path
import random
from typing import Any, Sequence

from voxtral.agent import LoggingMiddleware
from voxtral.log_utils import get_logger

import pytest
from unittest.mock import MagicMock

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
import langchain_core.messages as msgs
from langchain.agents import AgentState
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.tools import Tool


logger = get_logger()


@pytest.fixture
def middleware():
    return LoggingMiddleware(MagicMock())


SYSTEM_PROMPTS = [msgs.SystemMessage(content="foo"), None]
HUMAN_PROMPTS = [msgs.HumanMessage(content="foo"), None]
ANSWERS = [
    [
        msgs.AIMessage(
            content="",
            tool_calls=[msgs.ToolCall(name="foo", args={"foo": "bar"}, id="tool_foo")],
        ),
        msgs.ToolMessage(content="foo", tool_call_id="tool_foo"),
        msgs.AIMessage(content="bar"),
    ],
]
TOOLS = [[], [Tool(name="foo", description="bar", func=lambda x: None)]]
MESSAGES = [
    [],
    [
        msgs.AIMessage(
            content="",
            tool_calls=[
                msgs.ToolCall({"name": "test", "args": {"x": "bar"}, "id": "test"})
            ],
        ),
        msgs.ToolMessage(content="bar", tool_call_id="foo"),
        msgs.AIMessage(content="foo"),
    ],
]


def generate_perms[T](args: list[Sequence[T]], names: list[str]) -> dict[str, Any]:
    return {
        name: arg[random.randint(0, len(arg) - 1)]
        for name, arg in zip(names, args, strict=True)
    }


@pytest.mark.parametrize(
    "test_state",
    [
        generate_perms(
            [SYSTEM_PROMPTS, HUMAN_PROMPTS, MESSAGES, ANSWERS],
            ["sys_p", "h_p", "messages", "answ"],
        )
    ],
)
def test_middleware_collects_right_messages(
    test_state,
    middleware: LoggingMiddleware,
):
    # GIVEN
    test_state["messages"].append(test_state["h_p"])
    test_state["messages"].extend(test_state["answ"])
    state = AgentState({"messages": test_state["messages"]})

    # WHEN
    middleware.wrap_model_call(
        ModelRequest(
            model=GenericFakeChatModel(messages=iter([])),
            messages=test_state["messages"],
            system_message=test_state["sys_p"],
        ),
        handler=lambda req: ModelResponse([]),
    )
    middleware.after_agent(state, MagicMock())

    # THEN
    assert middleware._curr_sys_prompt == test_state["sys_p"]
    assert middleware._curr_prompt == test_state["h_p"]
    assert middleware._curr_answer == test_state["answ"]


@pytest.mark.parametrize(
    "state",
    [
        generate_perms(
            [SYSTEM_PROMPTS, HUMAN_PROMPTS, ANSWERS, TOOLS],
            ["sys_p", "h_p", "answ", "tools"],
        )
        for _ in range(10)
    ],
)
def test_write_convo_to_file(state: dict[str, Any]):
    # FileUT
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as temp_f:
        middleware = LoggingMiddleware(filepath=Path(temp_f.name))

    try:
        # GIVEN
        middleware._curr_answer = state["answ"]
        middleware._curr_prompt = state["h_p"]
        middleware._curr_sys_prompt = state["sys_p"]
        middleware._curr_available_tools = state["tools"]

        # WHEN
        middleware._save_to_file()

        # THEN
        with open(temp_f.name, "r") as file_ut:
            if not state["h_p"]:
                assert not file_ut.read()
            else:
                for line in file_ut:
                    assert json.loads(line)
    finally:
        temp_f.close()
