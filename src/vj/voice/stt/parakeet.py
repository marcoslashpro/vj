from typing import Any, Callable, Literal

import numpy as np
from vj.voice.base import BaseSTT
from vj.log_utils import get_logger

import onnx_asr


logger = get_logger()


PARAKEET_MODEL = Literal[
    "nemo-parakeet-ctc-0.6b",
    "nemo-parakeet-rnnt-0.6b",
    "nemo-parakeet-tdt-0.6b-v2",
    "nemo-parakeet-tdt-0.6b-v3",
]
QUANT = Literal[None, "int8", "int4"]


class ParakeetSTT(BaseSTT):
    def __init__(
        self, model: PARAKEET_MODEL, quant: QUANT
    ) -> None:
        super().__init__()
        self.model = onnx_asr.load_model(model, None, quantization=quant)

    def transcribe(self, audio: np.ndarray[Any, np.dtype[Any]], **params) -> str:
        "params are unused"
        prompt = self.model.recognize(audio)
        logger.debug(f"Transcribed: {prompt}")
        return prompt
