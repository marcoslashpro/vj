import onnxruntime as onrt
import numpy as np
from silero_vad import load_silero_vad

from pathlib import Path
from abc import ABC, abstractmethod

import torch

from vj.voice.base import VAD
from vj.log_utils import get_logger


logger = get_logger()


class SileroVAD(VAD):
    def __init__(self, threshold: float = 0.4) -> None:
        self._vad = load_silero_vad(onnx=True)
        self.threshold = threshold

    def _preprocess_audio(self, indata: np.ndarray) -> torch.Tensor:
        audio = indata.copy().flatten().astype(np.float32)
        # Center the waveform at 0
        audio -= np.mean(audio)
        peak = np.max(np.abs(audio))

        if peak > 0.01:
            audio = audio / peak

        return torch.from_numpy(audio).type(torch.float32)

    def is_speech(self, audio: np.ndarray, samplerate: int) -> bool:
        out = self._vad(self._preprocess_audio(audio), samplerate)
        prob: float = out[0][0]  # type: ignore
        return prob >= self.threshold
