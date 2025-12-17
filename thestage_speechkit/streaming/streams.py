import numpy as np
import sys
import time
import threading
from typing import Optional
from sounddevice import InputStream
from librosa import load, resample

import numpy as np

# ------------------------------
# FileStream
# ------------------------------


class ArrayStream:
    """
    Stream audio chunks from a numpy array with optional real-time pacing.

    Returns np.float32 mono chunks in [-1, 1], sampled at `sample_rate`.
    - If `real_time=False`: each next_chunk() returns exactly step_size_s seconds.
    - If `real_time=True`:
        * If elapsed < step -> wait remaining time and return step-size chunk.
        * If elapsed >= step -> return chunk sized (elapsed + step) seconds.
    """

    def __init__(
        self,
        audio_data: np.ndarray,
        step_size_s: float = 0.5,
        sample_rate: int = 16000,
        real_time: bool = True,
    ):
        self.audio_data = audio_data.flatten()
        self.sample_rate = sample_rate
        self.step_size_s = step_size_s
        self.real_time = real_time
        self._current_position = 0
        self._last_call_t = None
        self._eof = False

    def next_chunk(self) -> Optional[np.ndarray]:
        if self._eof:
            return None

        if not self.real_time:
            chunk_size = int(self.step_size_s * self.sample_rate)
            chunk = self.audio_data[
                self._current_position : self._current_position + chunk_size
            ]
            self._current_position += chunk_size
        else:
            elapsed = (
                None if self._last_call_t is None else (time.time() - self._last_call_t)
            )

            if elapsed is None:
                chunk_size = int(self.step_size_s * self.sample_rate)
            else:
                if elapsed > self.step_size_s:
                    chunk_size = int((elapsed + self.step_size_s) * self.sample_rate)
                else:
                    time.sleep(self.step_size_s - elapsed)
                    chunk_size = int(self.step_size_s * self.sample_rate)

            chunk = self.audio_data[
                self._current_position : self._current_position + chunk_size
            ]
            self._current_position += chunk_size
            self._last_call_t = time.time()

        if self._current_position >= len(self.audio_data):
            self._eof = True

        return chunk.astype(np.float32, copy=False)

    def close(self):
        self._current_position = 0
        self._last_call_t = None
        self._eof = False
        self.audio_data = None


class FileStream(ArrayStream):
    """
    Stream audio chunks from a WAV file with optional real-time pacing.

    Returns np.float32 mono chunks in [-1, 1], sampled at `sample_rate`.
    - If `real_time=False`: each next_chunk() returns exactly step_size_s seconds.
    - If `real_time=True`:
        * If elapsed < step -> wait remaining time and return step-size chunk.
        * If elapsed >= step -> return chunk sized (elapsed + step) seconds.
    """

    def __init__(
        self,
        path: str,
        step_size_s: float = 0.5,
        sample_rate: int = 16000,
        real_time: bool = True,
    ):
        self.path = path
        audio_data, sr = load(path, sr=sample_rate)
        if sr != sample_rate:
            audio_data = resample(audio_data, orgi_sr=sr, target_sr=sample_rate)
        super().__init__(audio_data, step_size_s, sample_rate, real_time)


# ------------------------------
# MicStream
# ------------------------------


class MicStream:
    """
    Real-time microphone stream.
    """

    def __init__(
        self,
        step_size_s: float = 0.5,
        sample_rate: int = 16000,
        device: Optional[int] = None,
        channels: int = 1,
    ):
        self.step_size_s = step_size_s
        self.sample_rate = sample_rate
        self.device = device
        self.channels = channels

        self.stream = InputStream(
            samplerate=sample_rate,
            blocksize=int(step_size_s * sample_rate / 2),
            device=device,
            channels=channels,
        )
        self.read_thread = threading.Thread(target=self.read_stream)
        self.queue = []
        self.last_chunk_time = None

    def read_stream(self):
        self.stream.start()
        while True:
            chunk, _ = self.stream.read(int(self.step_size_s * self.sample_rate))
            self.queue.append(chunk.squeeze())

    def next_chunk(self) -> Optional[np.ndarray]:
        if not self.read_thread.is_alive():
            self.read_thread.start()

        while not self.queue:
            time.sleep(0.01)
        chunk = np.concatenate(self.queue, axis=0)
        self.queue = []

        return chunk

    def close(self):
        self.stream.stop()
        self.stream.close()
        if self.read_thread.is_alive():
            self.read_thread.join(timeout=1.0)

    def __del__(self):
        self.close()


# ------------------------------
# StdoutStream
# ------------------------------


class StdoutStream:
    """
    Stream approved and assumption words to stdout.
    """

    def __init__(self):
        self.hide_cursor = "\x1b[?25l"
        self.show_cursor = "\x1b[?25h"
        self.clear_to_eol = "\x1b[K"
        self.prev_uncommitted_str = ""
        self.committed_str = ""
        sys.stdout.write(self.hide_cursor)
        self.approved_words = []

    def write(self, approved: list[str], assumption: list[str]):
        approved = [token["text"] for token in approved]
        assumption = [token["text"] for token in assumption]
        self.approved_words.extend(approved)

        if not assumption and not approved:
            return

        new_committed_str = "".join(self.approved_words)
        new_uncommitted_str = "".join(assumption)

        if self.prev_uncommitted_str:
            sys.stdout.write("\b" * len(self.prev_uncommitted_str))
            sys.stdout.write(self.clear_to_eol)

        if new_committed_str.startswith(self.committed_str):
            delta = new_committed_str[len(self.committed_str) :]
            if delta:
                sys.stdout.write(delta)
        else:
            sys.stdout.write("\r" + new_committed_str)

        self.committed_str = new_committed_str

        if new_uncommitted_str:
            sys.stdout.write(new_uncommitted_str)
            self.prev_uncommitted_str = new_uncommitted_str
        else:
            self.prev_uncommitted_str = ""

        sys.stdout.flush()

    def close(self):
        if self.prev_uncommitted_str:
            sys.stdout.write("\b" * len(self.prev_uncommitted_str) + self.clear_to_eol)
        sys.stdout.write("\n" + self.show_cursor)
        sys.stdout.flush()
        self.approved_words = []
