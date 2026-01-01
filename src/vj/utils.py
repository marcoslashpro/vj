from typing import Sequence, cast
from datetime import datetime

from langchain_core.messages import BaseMessage, ToolCall
from langchain_core.tools import (
    BaseTool,
    render_text_description_and_args,
)

MONTH_MAP = {
    "1": "January",
    "2": "February",
    "3": "March",
    "4": "April",
    "5": "May",
    "6": "June",
    "7": "July",
    "8": "August",
    "9": "September",
    "10": "October",
    "11": "November",
    "12": "December",
}
DAYS_MAP = {
    "1": "1st",
    "2": "2nd",
    "3": "3rd",
    "4": "4th",
    "5": "5th",
    "6": "6th",
    "7": "7th",
    "8": "8th",
    "9": "9th",
    "10": "10th",
    "11": "11th",
    "12": "12th",
    "13": "13th",
    "14": "14th",
    "15": "15th",
    "16": "16th",
    "17": "17th",
    "18": "18th",
    "19": "19th",
    "20": "20th",
    "21": "21st",
    "22": "22nd",
    "23": "23rd",
    "24": "24th",
    "25": "25th",
    "26": "26th",
    "27": "27th",
    "28": "28th",
    "29": "29th",
    "30": "30th",
    "31": "31st",
}


def add_few_shots(
    messages: list[BaseMessage], few_shots: list[BaseMessage]
) -> list[BaseMessage]:
    messages = messages + few_shots
    return messages


def generate_date_and_time() -> str:
    now = datetime.now()
    return f"Today is the {DAYS_MAP[str(now.day)]} of {MONTH_MAP[str(now.month)]}, {now.year}. The time is {now.strftime('%H:%M:%S')}"


def generate_tools_instruction(tools: Sequence[BaseTool] | None) -> str:
    if not tools:
        return "You have no tools available."
    else:
        return f"# Tools:\nThese are the tools available for you to use: {render_text_description_and_args(cast(list, tools))}"


def render_tool_call_description(tool_calls: list[ToolCall]) -> str:
    final = ""
    for call in tool_calls:
        final += f"""
function {call['name']} with arguments: {call['args']}
"""
    return final
