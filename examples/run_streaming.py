from thestage_speechkit.streaming import (
    StreamingPipeline,
    MicStream,
    FileStream,
    StdoutStream,
)
from transformers.utils import logging as hf_logging
import logging, warnings

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use-mic", action="store_true")
parser.add_argument(
    "--step-size",
    type=float,
    default=0.05,
    help="Size of small audio chunks in seconds fed into the streaming pipeline",
)
parser.add_argument(
    "--process-window",
    type=float,
    default=0.5,
    help="Minimum accumulated audio (in seconds) before running ASR processing",
)
parser.add_argument(
    "--language",
    type=str,
    default='en',
    help="Language of the audio",
)
parser.add_argument(
    "--audio-file",
    type=str,
    default='example_speech.wav',
    help="Path to the audio file to transcribe",
)
args = parser.parse_args()

# Silence Transformers logs
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
logging.getLogger("transformers").setLevel(logging.ERROR)
# Silence specific runtime UserWarnings
warnings.filterwarnings(
    "ignore",
    message=r"Whisper did not predict an ending timestamp.*",
    category=UserWarning,
)

streaming_model = StreamingPipeline(
    model='TheStageAI/thewhisper-large-v3-turbo',
    chunk_length_s=10,
    platform='apple',
    language=args.language,
    min_process_chunk_s=args.process_window,
    # revision='15d196cdee60d9ff98b7eadb549aa2566e83218c'
)

if args.use_mic:
    audio_stream = MicStream(step_size_s=args.step_size)
else:
    audio_stream = FileStream(args.audio_file, step_size_s=args.step_size)

full_approved_text = ""
green = "\033[92m"
yellow = "\033[93m"
reset = "\033[0m"

while True:
    chunk = audio_stream.next_chunk()
    if chunk is not None:
        approved, assumption = streaming_model(chunk)

        approved_text = "".join([token['text'] for token in approved])
        assumption_text = "".join([token['text'] for token in assumption])
        full_approved_text += approved_text
        # Print approved text in green and assumption text in yellow
        if approved_text or assumption_text:
            output = ""
            if full_approved_text:
                output += f"{green}{full_approved_text}{reset}"
            if assumption_text:
                if approved_text:
                    output += " "
                output += f"{yellow}{assumption_text}{reset}"
            print(output)
    else:
        break

audio_stream.close()
