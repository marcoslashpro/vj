from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class VAD(ABC):
    @abstractmethod
    def is_speech(self, audio: np.ndarray, samplerate: int) -> bool:
        """Business logic to infer if an audio chunk is speech or not"""


class WWD(ABC):
    def __init__(self, ww_duration_ms: int) -> None:
        self.ww_duration_ms = ww_duration_ms

    @abstractmethod
    def predict(self, audio: np.ndarray[Any, np.dtype[np.float32]]) -> bool:
        pass

class BaseSTT(ABC):
    sample_rate: int = 16000

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        **params,
    ) -> str:
        pass