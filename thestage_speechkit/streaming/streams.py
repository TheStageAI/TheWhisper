import sys
import time
import math
import threading
import queue
from typing import Optional
from librosa import load, resample

import numpy as np

# ------------------------------
# FileStream
# ------------------------------

class FileStream:
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
        self.sample_rate = sample_rate
        self.step_size_s = step_size_s
        self.real_time = real_time
        
        self.audio_data, sr = load(path, sr=sample_rate)
        if sr != sample_rate:
            self.audio_data = resample(self.audio_data, orgi_sr=sr, target_sr=sample_rate)
        self.audio_data = self.audio_data.flatten()
        self._current_position = 0
        self._last_call_t = None
        self._eof = False

    def next_chunk(self) -> Optional[np.ndarray]:
        if self._eof:
            return None

        if not self.real_time:
            chunk_size = int(self.step_size_s * self.sample_rate)
            chunk = self.audio_data[self._current_position:self._current_position + chunk_size]
            self._current_position += chunk_size
        else:
            elapsed = None if self._last_call_t is None else (time.time() - self._last_call_t)

            if elapsed is None:
                chunk_size = int(self.step_size_s * self.sample_rate)
            else:
                if elapsed > self.step_size_s:
                    chunk_size = int((elapsed + self.step_size_s) * self.sample_rate)
                    # print(f"Chunk size: {chunk_size}", "elapsed: ", elapsed)
                else:
                    time.sleep(self.step_size_s - elapsed)
                    chunk_size = int(self.step_size_s * self.sample_rate)

            chunk = self.audio_data[self._current_position:self._current_position + chunk_size]
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


# ------------------------------
# MicStream
# ------------------------------

class MicStream:
    """
    Real-time microphone stream.

    Always real-time (no real_time=False mode). Returns np.float32 mono chunks
    in [-1, 1] at the given `sample_rate`.

    The same pacing logic is applied:
      - If elapsed < step -> wait remaining time then return ~step seconds.
      - If elapsed >= step -> return chunk sized (elapsed + step) seconds.
    """
    def __init__(
        self,
        step_size_s: float = 0.5,
        sample_rate: int = 16000,
        device: Optional[int] = None,
        channels: int = 1,
    ):
        try:
            import sounddevice as sd
        except ImportError as e:
            raise ImportError(
                "MicStream requires the 'sounddevice' package. Install with 'pip install sounddevice'."
            ) from e

        self.sample_rate = int(sample_rate)
        self.step_size_s = float(step_size_s)
        self.channels = int(channels)
        if self.channels != 1:
            # We'll capture multi-channel but downmix to mono
            pass

        self._q = queue.Queue(maxsize=64)
        self._buf = np.zeros(0, dtype=np.float32)
        self._last_call_t: Optional[float] = None
        self._closed = False

        def callback(indata, frames, time_info, status):
            if status:
                # You could log warnings here if desired
                pass
            # indata is float32, shape (frames, channels)
            block = indata.astype(np.float32, copy=False)
            if self.channels > 1:
                block = block.mean(axis=1)
            else:
                block = block.reshape(-1)
            try:
                self._q.put_nowait(block.copy())
            except queue.Full:
                # Drop if overwhelmed
                pass

        self._sd = sd
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            device=device,
            channels=self.channels,
            dtype="float32",
            blocksize=int(self.step_size_s * self.sample_rate / 2),
            callback=callback,
        )
        self._stream.start()

    def _desired_samples(self, elapsed: Optional[float]) -> int:
        # Always real-time
        if elapsed is None:
            duration = self.step_size_s
        elif elapsed < self.step_size_s:
            time.sleep(self.step_size_s - elapsed)
            duration = self.step_size_s
        else:
            duration = elapsed + self.step_size_s
        return int(round(duration * self.sample_rate))

    def next_chunk(self) -> np.ndarray:
        if self._closed:
            return np.zeros(0, dtype=np.float32)

        now = time.monotonic()
        elapsed = None if self._last_call_t is None else (now - self._last_call_t)

        need = self._desired_samples(elapsed)

        # Fill internal buffer from queue until enough samples
        while self._buf.size < need and not self._closed:
            try:
                block = self._q.get(timeout=0.1)
                self._buf = np.concatenate([self._buf, block]) if self._buf.size else block
            except queue.Empty:
                # Keep waiting until we can satisfy `need` or until closed
                continue

        if self._buf.size == 0:
            # Nothing captured (e.g., muted device). Return silence of the desired size.
            self._last_call_t = time.monotonic()
            return np.zeros(need, dtype=np.float32)

        if self._buf.size <= need:
            out = self._buf
            # If still short, pad with zeros to meet `need`
            if out.size < need:
                pad = np.zeros(need - out.size, dtype=np.float32)
                out = np.concatenate([out, pad]) if out.size else pad
            self._buf = np.zeros(0, dtype=np.float32)
            self._last_call_t = time.monotonic()
            return out.astype(np.float32, copy=False)

        out = self._buf[:need]
        self._buf = self._buf[need:]
        self._last_call_t = time.monotonic()
        return out.astype(np.float32, copy=False)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        # Drain the queue
        with self._q.mutex:
            self._q.queue.clear()


class StdoutStream:
    def __init__(self):
        self.hide_cursor = "\x1b[?25l"
        self.show_cursor = "\x1b[?25h"
        self.clear_to_eol = "\x1b[K"
        self.prev_uncommitted_str = ""
        self.committed_str = ""
        sys.stdout.write(self.hide_cursor)

    def write(self, approved: list[str], assumption: list[str]):
        approved = [token['text'].strip() for token in approved]
        assumption = [token['text'].strip() for token in assumption]

        new_committed_str = " ".join(approved)
        new_uncommitted_str = " ".join(assumption)

        if self.prev_uncommitted_str:
            sys.stdout.write("\b" * len(self.prev_uncommitted_str))
            sys.stdout.write(self.clear_to_eol)

        if new_committed_str.startswith(self.committed_str):
            delta = new_committed_str[len(self.committed_str):]
            if delta:
                sys.stdout.write(delta)
        else:
            sys.stdout.write("\r" + new_committed_str)

        self.committed_str = new_committed_str

        if new_uncommitted_str and self.committed_str:
            prefix = " "
        else:
            prefix = ""

        if new_uncommitted_str:
            sys.stdout.write(prefix + new_uncommitted_str)
            self.prev_uncommitted_str = prefix + new_uncommitted_str
        else:
            self.prev_uncommitted_str = ""

        sys.stdout.flush()

    def close(self):
        if self.prev_uncommitted_str:
            sys.stdout.write("\b" * len(self.prev_uncommitted_str) + self.clear_to_eol)
        sys.stdout.write("\n" + self.show_cursor)
        sys.stdout.flush()
