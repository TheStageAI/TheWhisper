from librosa import load, resample
from thestage_speechkit.nvidia import ASRPipeline

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--audio-file",
    type=str,
    default='example_speech.wav',
    help="Path to the audio file to transcribe",
)
args = parser.parse_args()

generate_kwargs = {
    "num_beams": 1,
    "do_sample": False,
    "use_cache": True,
    "language": "en",
}
chunk_length_s = 10
pipe = ASRPipeline(
    "TheStageAI/thewhisper-large-v3-turbo",
    chunk_length_s=chunk_length_s,
    model_size="S",
    batch_size=32,
    device="cuda",
)
audio, sr = load(args.audio_file)
audio = resample(audio, orig_sr=sr, target_sr=16000)
output = pipe(
    audio,
    generate_kwargs=generate_kwargs,
    chunk_length_s=chunk_length_s - 1,
)
print(output)
