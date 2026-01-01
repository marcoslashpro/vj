import wikipedia
from langchain_core.tools import Tool

from vj.log_utils import get_logger


logger = get_logger()


def think(text: str) -> None:
    logger.debug(f"VJ called think tool with: {text}")
    return


def calc(expr: str) -> str:
    logger.debug(f"VJ called calc tool with: {expr}")
    try:
        res = eval(expr)
        logger.debug(f"Evaluated expression: {res}")
        return res
    except Exception as e:
        msg = f"Invalid expression {expr}, full error: {e}"
        logger.error(f"{msg}")
        return msg


def wiki(topic: str) -> str:
    logger.debug(f"VJ called wiki tool with: {topic}")
    try:
        summary = wikipedia.summary(topic)
        logger.debug(f"Response from wiki tool: {summary}")
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return str(e) + "\nChoose the one most relevant to the current topic."
    except wikipedia.exceptions.PageError as e:
        return str(e)


tools = [
    Tool(
        name="think",
        func=think,
        description="Use this tool as a scratchpad where to write your thoughts.",
    ),
    Tool(
        name="calculator",
        func=calc,
        description="Use this tool in order to evaluate a mathematical expression. The 'expr' argument must a valid python expression.",
    ),
    Tool(
        name="search_wiki",
        func=wiki,
        description=(
            "Use this tool as an interface with wikipedia, it returns summaries extracted on Wikipedia of the given topic.\n"
            "If tool the fails in providing a valuable response or if it fails for another external reason, "
            "then you can confidently tell to the user that you do not know the answer to the question.\n"
            "If you get an answer from this tool, summarize the results to the user in just a couple of sentences."
        ),
    ),
]
