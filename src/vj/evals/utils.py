from datetime import datetime
import json
from pathlib import Path
import queue

import numpy as np
import librosa

from vj.log_utils import get_logger
from vj import ASSETS_DIR


logger = get_logger()


RESULTS_DIR = Path(__file__).parent / "results"


def load_audio(
    file: Path, samplerate: int = 16000, mono: bool = True, dtype=np.float32
) -> np.ndarray:
    audio, _ = librosa.load(file, sr=samplerate, mono=mono, dtype=dtype)
    return audio


def chunk_audio(audio: np.ndarray, chunk_size: int = 512) -> queue.Queue[np.ndarray]:
    q: queue.Queue[np.ndarray] = queue.Queue()
    buff_size = len(audio) // chunk_size
    for buff in np.array_split(audio, buff_size):
        q.put(buff)

    return q


def save_evals(results_dir: Path, scores: list[dict]) -> None:
    final_path = results_dir / f"evaluations.json"
    with final_path.open("a") as f:
        try:
            f.write(json.dumps(scores, indent=4))
        except Exception as e:
            logger.error(f"While dumping JSON: {e}")
            raise
