from typing import Any
import numpy as np

from vj.voice.base import BaseSTT
from vj.log_utils import get_logger

from pywhispercpp.model import Model, Segment


logger = get_logger()


class WhisperSTT(BaseSTT):
    def __init__(
        self,
        model: str = "small",
        n_threads: int = 16,
        no_context: bool = False,
        beam_size: int = 4,
        patience: int = -1,
        greedy_best_of: int = 2,
        print_progress: bool = False,
    ) -> None:
        beam_search = {"beam_size": beam_size, "patience": patience}
        greedy = {"best_of": greedy_best_of}

        self.pwcpp = Model(
            model,
            n_threads=n_threads,
            no_context=no_context,
            print_progress=print_progress,
            beam_search=beam_search,
            greedy=greedy,
        )

    def transcribe(self, audio: np.ndarray[Any, np.dtype[np.float32]], **params) -> str:
        segs = self.pwcpp.transcribe(audio, **params)
        return self._new_segment_callback(segs)

    def _new_segment_callback(self, segs: list[Segment]) -> str:
        logger.debug(f"Got back segments: {segs}")
        return " ".join([seg.text for seg in segs])
