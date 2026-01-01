import time
from typing import Any, Callable
import multiprocessing as mp
from multiprocessing import shared_memory, sharedctypes, synchronize
import queue
from abc import ABC, abstractmethod
from collections import deque

import sounddevice as sd
import numpy as np

from vj.voice.base import BaseSTT, VAD, WWD
from vj.log_utils import get_logger


logger = get_logger()

SAMPLE_RATE = 16000
BLOCK_SIZE = 512


class BaseAssistant(ABC):
    def __init__(
        self,
        vad: VAD,
        stt: BaseSTT,
        callback_fn: Callable[[str], None],
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        input_device: int | None = None,
        silence_threshold: int = 20,
        q_threshold: int = 5,
        dtype: Any = np.float32,
    ):
        # Pipeline
        self.stt = stt
        self.vad = vad
        self.callback_fn = callback_fn  # function to execute upon STT transcription

        # VAD detection logic and turn taking
        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self.q_threshold = q_threshold
        self.silence_threshold = silence_threshold

        # Stream related params
        self.block_size = block_size  # frames dim
        self.dtype = dtype  # audio dtype
        self.input_device = input_device  # input recording device
        self.sample_rate = sample_rate  # input audio samplerate

        # Internal silence state
        self._silence_counter: int = 0

    def start(self) -> None:
        """
        Starts audio input detection.
        Calls `_setup()` before entering the stream and `_cleanup()` on KeyboardInterrupt
        """
        self._setup()
        with sd.InputStream(
            device=self.input_device,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            callback=self._audio_callback,
            dtype=self.dtype,
        ):

            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self._cleanup()

    def _setup(self) -> None:
        """Startup function called before entering the stream, it is not required to implement and a call to super is not required."""
        pass

    def _cleanup(self) -> None:
        """Cleanup function called after a KeyboardInterrupt, it is not required to implement and a call to super is not required."""
        pass

    @abstractmethod
    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        """
        Method that will be executed on incoming audio frames.
        A call to super() asserts that `fames == self.block_size`.
        The subclass must call `on_speech` inside of this method in order to execute the
        callback function.
        """
        assert (
            frames == self.block_size
        ), f"Invalid number of input frames, {frames} != {self.block_size}"

    def on_speech(self, prompt: str) -> None:
        """Method that has to be called in order to execute the given callback_fn.
        If timed is given, the given callback_fn is timed for performance."""
        self.callback_fn(prompt)


class Assistant(BaseAssistant):
    def _audio_callback(self, indata: np.ndarray, frames: int, time, status):
        super()._audio_callback(indata, frames, time, status)
        is_speech = self.vad.is_speech(indata, self.sample_rate)

        if is_speech:
            self.q.put(indata.copy())
            self._silence_counter = 0
        else:
            if self._silence_counter >= self.silence_threshold:
                if self.q.qsize() > self.q_threshold:
                    self._transcribe_speech()
                    self._silence_counter = 0
            else:
                self._silence_counter += 1

    def _transcribe_speech(self, **extra_params) -> None:
        curr = []
        while self.q.qsize() > 0:
            curr.append(self.q.get())

        if len(curr) > 0:
            audio_data = np.concatenate(curr).flatten()

            self.on_speech(self.stt.transcribe(audio_data, **extra_params))


class LiveAssistant:
    def __init__(
        self,
        vad: VAD,
        wwd: WWD,
        stt: BaseSTT,
        agent,
        tts,
        stop_t: int,
        sample_rate: int = SAMPLE_RATE,
        input_device: int | None = None,
        block_size: int = BLOCK_SIZE,
        dtype: Any = np.float32,
    ) -> None:
        self.vad = vad
        self.wwd = wwd
        self.stt = stt
        self.agent = agent
        self.tts = tts

        self.stop_t = stop_t  # used to identify EOS

        self._silence_t = 0  # time-step silence counter

        # Stream related params
        self.block_size = block_size  # frames dim
        self.dtype = dtype  # audio dtype
        self.input_device = input_device  # input recording device
        self.sample_rate = sample_rate  # input audio samplerate

        # Internal speaking state
        self._user_is_speaking: sharedctypes.Synchronized[int] = mp.Value("I", 0)
        self._sys_is_speaking: sharedctypes.Synchronized[int] = mp.Value("I", 0)
        self._is_final: sharedctypes.Synchronized[bool] = mp.Value("b", False)

        # Incoming data from the stream to hand over to VAD
        self._audio_frames_q: deque = deque(
            maxlen=round(wwd.ww_duration_ms / self.block_size) * 20
        )
        # Q to send VAD detected speech frames to STT
        self._speech_frames_q: mp.Queue[np.ndarray[Any, np.dtype[np.float32]]] = (
            mp.Queue()
        )
        # Synchronized Q of transcriptions used between STT and Agent nodes
        self._trascribed_q: mp.Queue[str] = mp.Queue()
        # Q to send LLM completions to the TTS
        self._completions_q: mp.Queue[str] = mp.Queue()
        # Pipe to handle speech communication between main process and TTS process
        self._main_commm_pipe, self._tts_comm_pipe = mp.Pipe()

    def start(self) -> None:
        transcription_proc = mp.Process(
            None,
            self.run_stt,
            args=(self._speech_frames_q, self._is_final),
            daemon=True,
        )
        agent_proc = mp.Process(
            None, self.run_agent, args=(self._trascribed_q,), daemon=True
        )
        transcription_proc.start()
        agent_proc.start()

        with sd.InputStream(
            device=self.input_device,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            callback=self._audio_callback,
            dtype=self.dtype,
        ):
            try:
                while True:
                    time.sleep(0.1)
            finally:
                transcription_proc.kill()
                agent_proc.kill()

    def _audio_callback(
        self, indata: np.ndarray[Any, np.dtype[np.float32]], frames: int, time, status
    ) -> None:
        assert (
            frames == self.block_size
        ), f"Invalid number of input frames, {frames} != {self.block_size}"
        if status:
            logger.warning(f"Audio callback status: {status}")
        self._audio_frames_q.append(indata.copy())

        if self.vad.is_speech(indata, self.sample_rate):
            if self.wwd.predict(np.concatenate(self._audio_frames_q).flatten()):
                logger.debug(f"WW detected")
                with self._user_is_speaking.get_lock():
                    self._user_is_speaking.value = True
                    # reset silence counter
                    self._silence_t = 0
                    # clear frames q
                    self._audio_frames_q.clear()

            # Only pass to the STT speech after the user has said the WW
            with self._user_is_speaking.get_lock():
                if self._user_is_speaking.value:
                    self._speech_frames_q.put(indata)
        else:
            self._silence_t += 1

            if self._silence_t == self.stop_t:
                with self._is_final.get_lock():
                    self._is_final.value = True
                with self._user_is_speaking.get_lock():
                    self._user_is_speaking.value = False

    def run_stt(
        self,
        speech_frames_q: queue.Queue[np.ndarray[Any, np.dtype[np.float32]]],
        is_final: sharedctypes.Synchronized,
    ) -> None:
        curr_frames: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        transcribed = None
        while True:
            if speech_frames_q.qsize() == 0 and not is_final.value:
                time.sleep(0.1)
                continue

            while speech_frames_q.qsize() > 0:
                curr_frames.append(speech_frames_q.get())

            if curr_frames:
                audio_frames = np.concatenate(curr_frames).flatten()
                transcribed = self.stt.transcribe(audio_frames)

            with is_final.get_lock():
                if is_final.value:
                    if transcribed:
                        self._trascribed_q.put(transcribed)
                    transcribed = ""
                    curr_frames.clear()
                    is_final.value = False

    def run_agent(self, transcribed: queue.Queue[str]) -> None:
        while True:
            speech = transcribed.get()
            logger.debug(f"Received: {speech}")
