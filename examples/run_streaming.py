from thestage_speechkit.streaming import (
    StreamingPipeline, MicStream, FileStream, StdoutStream
)
from transformers.utils import logging as hf_logging
import logging, warnings

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use-mic', action='store_true')
args = parser.parse_args()

# Silence Transformers logs
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
logging.getLogger("transformers").setLevel(logging.ERROR)
# Silence specific runtime UserWarnings
warnings.filterwarnings("ignore", message=r"Whisper did not predict an ending timestamp.*", category=UserWarning)

streaming_model = StreamingPipeline(
    model='TheStageAI/thewhisper-large-v3-turbo',
    chunk_length_s=15,
    platform='apple',
    language='en',
)

if args.use_mic:
    audio_stream = MicStream()
else:
    audio_stream = FileStream('example_speech.wav')

output_stream = StdoutStream()

while True:
    chunk = audio_stream.next_chunk()
    if chunk is not None:
        approved, assumption = streaming_model(chunk)
        output_stream.write(approved, assumption)
    else:
        break

audio_stream.close()
output_stream.close()
