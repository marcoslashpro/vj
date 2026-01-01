from datetime import datetime, timezone
import logging
from typing import Literal, Mapping, Any, TypedDict
import json


class JSONFormatter(logging.Formatter):
    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: Literal["%"] | Literal["{"] | Literal["$"] = "%",
        validate: bool = True,
        *,
        defaults: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(self._prepare_log_msg(record))

    def _prepare_log_msg(self, record: logging.LogRecord) -> dict[str, Any]:
        msg = {
            "level": record.levelname,
            "message": record.getMessage(),
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
        }
        if record.exc_info:
            msg["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            msg["stack_info"] = self.formatStack(record.stack_info)

        return msg

class QA(TypedDict):
    question: str
    steps: list[Any]
    answer: str
class Convo(TypedDict):
    sys_prompt: str
    messages: list[QA]
JSON_CONTENT = list[Convo]