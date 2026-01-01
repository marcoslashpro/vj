# coding=utf-8
# Copyright 2023 The Google Research Authors.
# Modifications copyright 2024 Kevin Ahrendt.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from ai_edge_litert.interpreter import Interpreter
from pymicro_features import MicroFrontend

from vj.voice.base import WWD
from vj.log_utils import get_logger


logger = get_logger()


class HeyJarvis(WWD):
    """
    Class for loading and running tflite microwakeword models

    Args:
        tflite_model_path (str): Path to tflite model file.
        stride (int | None, optional): Time dimension's stride. If None, then the stride is the input tensor's time dimension. Defaults to None.
    """

    def __init__(
        self,
        tflite_model_path: str,
        ww_duration_ms: int = 2000,
        max_window_size: int = 5,
        window_threshold: int = 5,
        token_threshold: float = 0.5,
        stride: int | None = None,
    ):
        super().__init__(ww_duration_ms)
        # Load tflite model
        interpreter = Interpreter(
            model_path=tflite_model_path,
        )
        interpreter.allocate_tensors()

        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()

        self.is_quantized_model = self.input_details[0]["dtype"] == np.int8
        self.input_feature_slices = self.input_details[0]["shape"][1]

        if stride is None:
            self.stride = self.input_feature_slices
        else:
            self.stride = stride

        for s in range(len(self.input_details)):
            if self.is_quantized_model:
                interpreter.set_tensor(
                    self.input_details[s]["index"],
                    np.zeros(self.input_details[s]["shape"], dtype=np.int8),
                )
            else:
                interpreter.set_tensor(
                    self.input_details[s]["index"],
                    np.zeros(self.input_details[s]["shape"], dtype=np.float32),
                )

        self.model = interpreter

        # Threshold for the probability of one single token
        self.token_threshold = token_threshold
        # Threshold for the probability of the entire window
        self.window_threshold = window_threshold
        self.max_window_size = max_window_size

        # Window of detected WW
        self._window = deque(maxlen=self.max_window_size)

    def predict(self, data: np.ndarray) -> bool:
        """Run the model on a single clip of audio data

        Args:
            data (numpy.ndarray): input data for the model (16 khz, 16-bit PCM audio data)

        Returns:
            list: model predictions for the input audio data
        """

        # Get the spectrogram
        spectrogram = self.generate_features_for_clip(data)

        pred = self.predict_spectrogram(spectrogram)
        if pred:
            high_conf_frames = np.sum(np.array(pred) > self.token_threshold)
            if high_conf_frames >= self.window_threshold:
                return True
        return False

    def generate_features_for_clip(
        self,
        audio_samples: np.ndarray,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Generates spectrogram features for the given audio data.

        Args:
            audio_samples (numpy.ndarray): The clip's audio samples.

        Raises:
            ValueError: If the provided audio data is not a 16-bit integer array.

        Returns:
            numpy.ndarray: The spectrogram features for the provided audio clip.
        """

        # Convert any float formatted audio data to an int16 array
        if audio_samples.dtype in (np.float32, np.float64):
            max_val = np.abs(audio_samples).max()
            if max_val > 0:
                # Scale to ~0.8 of full range to avoid clipping
                audio_samples = audio_samples * (0.8 / max_val)

            audio_samples = np.clip((audio_samples * 32768), -32768, 32767).astype(
                np.int16
            )

        audio_buff = audio_samples.tobytes()
        micro_frontend = MicroFrontend()
        features = []
        audio_idx = 0
        num_audio_bytes = len(audio_buff)
        while audio_idx + 160 * 2 < num_audio_bytes:
            frontend_result = micro_frontend.process_samples(
                audio_buff[audio_idx : audio_idx + 160 * 2]
            )
            audio_idx += frontend_result.samples_read * 2
            if frontend_result.features:
                features.append(frontend_result.features)

        spectrogram = np.array(features).astype(np.float32)
        return spectrogram

    def predict_spectrogram(
        self, spectrogram: np.ndarray
    ) -> list[np.ndarray[Any, np.dtype[np.float64]]]:
        """Run the model on a single spectrogram

        Args:
            spectrogram (numpy.ndarray): Input spectrogram.

        Returns:
            list: model predictions for the input audio data
        """

        # Spectrograms with type np.uint16 haven't been scaled
        if np.issubdtype(spectrogram.dtype, np.uint16):
            spectrogram = spectrogram.astype(np.float32) * 0.0390625
        elif np.issubdtype(spectrogram.dtype, np.float64):
            spectrogram = spectrogram.astype(np.float32)

        # Slice the input data into the required number of chunks
        chunks = []
        for last_index in range(
            self.input_feature_slices, len(spectrogram) + 1, self.stride
        ):
            chunk = spectrogram[last_index - self.input_feature_slices : last_index]
            if len(chunk) == self.input_feature_slices:
                chunks.append(chunk)

        # Get the prediction for each chunk
        predictions = []
        for chunk in chunks:
            if self.is_quantized_model and spectrogram.dtype != np.int8:
                chunk = self.quantize_input_data(chunk, self.input_details[0])

            self.model.set_tensor(
                self.input_details[0]["index"],
                np.reshape(chunk, self.input_details[0]["shape"]),
            )
            self.model.invoke()

            output = self.model.get_tensor(self.output_details[0]["index"])[0][0]
            if self.is_quantized_model:
                output = self.dequantize_output_data(output, self.output_details[0])

            predictions.append(output)

        return predictions

    def quantize_input_data(self, data: np.ndarray, input_details: dict) -> np.ndarray:
        """quantize the input data using scale and zero point

        Args:
            data (numpy.array in float): input data for the interpreter
            input_details (dict): output of get_input_details from the tflm interpreter.

        Returns:
          numpy.ndarray: quantized data as int8 dtype
        """
        # Get input quantization parameters
        data_type = input_details["dtype"]

        input_quantization_parameters = input_details["quantization_parameters"]
        input_scale, input_zero_point = (
            input_quantization_parameters["scales"][0],
            input_quantization_parameters["zero_points"][0],
        )
        # quantize the input data
        data = data / input_scale + input_zero_point
        return data.astype(data_type)

    def dequantize_output_data(
        self, data: np.ndarray, output_details: dict
    ) -> np.ndarray:
        """Dequantize the model output

        Args:
            data (numpy.ndarray): integer data to be dequantized
            output_details (dict): TFLM interpreter model output details

        Returns:
            numpy.ndarray: dequantized data as float32 dtype
        """
        output_quantization_parameters = output_details["quantization_parameters"]
        output_scale = 255.0  # assume (u)int8 quantization
        output_zero_point = output_quantization_parameters["zero_points"][0]
        # Caveat: tflm_output_quant need to be converted to float to avoid integer
        # overflow during dequantization
        # e.g., (tflm_output_quant -output_zero_point) and
        # (tflm_output_quant + (-output_zero_point))
        # can produce different results (int8 calculation)
        # return output_scale * (data.astype(np.float32) - output_zero_point)
        return 1 / output_scale * (data.astype(np.float32) - output_zero_point)
