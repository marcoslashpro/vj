from typing import Any, Callable, Literal
import numpy as np
from vj.voice.base import BaseSTT


MOONSHINE_MODEL = Literal["tiny", "base"]
MAX_MOONSHINE_AUDIO_LENGTH_S = 64
MIN_MOONSHINE_AUDIO_LENGTH_S = 0.1


class MoonshineSTT(BaseSTT):
    def __init__(
        self,
        model: MOONSHINE_MODEL = "base",
    ) -> None:
        try:
            import moonshine_onnx
            import tokenizers
        except ImportError:
            raise ImportError(
                "moonshine_onnx module not found. In order to use the MoonshineSTT please"
                " install it with `uv pip install git+https://****@github.com/moonshine-ai/moonshine.git@23c2df746a3a017224efe8ece2492588751afdfa#subdirectory=moonshine-onnx`"
            )

        self.model = moonshine_onnx.MoonshineOnnxModel(None, model)
        self.tokenizer: tokenizers.Tokenizer = moonshine_onnx.load_tokenizer()

        # Enable padding for sequences < 0.1 seconds
        self.tokenizer.enable_padding()

    def transcribe(
        self,
        audio: np.ndarray,
        **params,
    ) -> str:
        "params are unused"
        if not audio.shape == 2:
            audio = audio[None, ...]
        audio_len_s = audio.size / self.sample_rate

        audio_chunks: list[np.ndarray] | None = None
        if audio_len_s > MAX_MOONSHINE_AUDIO_LENGTH_S:
            audio_chunks = np.array_split(
                audio, round(audio_len_s / MAX_MOONSHINE_AUDIO_LENGTH_S)
            )

        if audio_chunks:
            results: list[str] = []
            for chunk in audio_chunks:
                results.extend(self.tokenizer.decode_batch(self.model.generate(chunk)))
            return " ".join(results)
        return self.tokenizer.decode_batch(self.model.generate(audio))[0]
