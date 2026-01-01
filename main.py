from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessageChunk, AIMessage
from langchain.agents import create_agent

from vj.assistant import LiveAssistant
from vj.voice.vad import SileroVAD
from vj.voice.stt import ParakeetSTT
from vj.voice.wwd import HeyJarvis
from vj.tools import tools
from vj.utils import generate_date_and_time, generate_tools_instruction
from vj.log_utils import get_logger
from vj.agent.middleware import LoggingMiddleware
from vj import ASSETS_DIR


logger = get_logger()


sys_prompt = (
    generate_date_and_time()
    + f"""
You are a model that can do function calling with the following functions"""
    + generate_tools_instruction(tools)
)
llm = ChatOllama(model="ministral-3:3b")
agent = create_agent(
    llm, tools=tools, system_prompt=sys_prompt, middleware=[LoggingMiddleware()]
)
convo = []


def run_llm(prompt: str) -> None:
    convo.append(HumanMessage(content=prompt))
    final = ""

    logger.debug(f"Running inference with messages: {convo}")
    try:
        for chunk, _ in agent.stream({"messages": convo}, stream_mode="messages"):
            if isinstance(chunk, AIMessageChunk):
                assert isinstance(chunk.content, str), f"{type(chunk.content)} != str"
                print(chunk.content, end="", flush=True)
                final += chunk.content
    except Exception as e:
        logger.error(f"Error during Agent response: {e}")

    convo.append(AIMessage(content=final))
    print("\nSpeak:")


tj = LiveAssistant(
    vad=SileroVAD(),
    wwd=HeyJarvis(str(ASSETS_DIR / "models/wwd.tflite")),
    stt=ParakeetSTT(model="nemo-parakeet-tdt-0.6b-v2", quant="int8"),
    agent=agent,
    tts=None,
    stop_t=20,
)


def main() -> None:
    logger.info(f"TJ's ready")
    tj.start()


if __name__ == "__main__":
    main()
