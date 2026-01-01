import sounddevice as sd
import wave
import numpy as np

from argparse import ArgumentParser

from voxtral import ASSETS_DIR


def main(args):
    block_size = 512
    framerate = 16000
    n_channels = 1
    in_stream = sd.InputStream(
        device=None,
        channels=n_channels,
        samplerate=framerate,
        blocksize=block_size,
    )
    audio: list[np.ndarray] = []

    try:
        print(f"Recording started, press CTRL + C to stop")
        in_stream.start()

        while True:
            audio.append(in_stream.read(block_size)[0])
    except:
        pass
    finally:
        if audio:
            with wave.open(str(ASSETS_DIR / args.filepath), "wb") as audio_f:
                audio_f.setsampwidth(2)
                audio_f.setnchannels(n_channels)
                audio_f.setframerate(framerate)
                recording = np.concatenate(audio)
                recording_int16 = (recording * 32767).astype(np.int16)
                audio_f.writeframes(recording_int16.tobytes())
                print(f"\nRecording saved to: {ASSETS_DIR / args.filepath}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filepath")
    args = parser.parse_args()
    main(args)
