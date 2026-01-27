from librosa import load, resample
from thestage_speechkit.apple import ASRPipeline

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--audio-file",
    type=str,
    default='example_speech.wav',
    help="Path to the audio file to transcribe",
)
parser.add_argument(
    "--language",
    type=str,
    default='en',
    help="Language of the audio",
)
args = parser.parse_args()

chunk_length_s = 10
pipe = ASRPipeline(
    "TheStageAI/thewhisper-large-v3-turbo",
    chunk_length_s=chunk_length_s,
    model_size="S",
    # revision='1b649dccd5944ef5a38ade18cae1c2d5ead144f2'
)
generate_kwargs = {
    "max_new_tokens": 128,
    "num_beams": 1,
    "do_sample": False,
    "use_cache": True,
    'language': args.language,
    'task': 'transcribe',
}
audio, sr = load(args.audio_file)
audio = resample(audio, orig_sr=sr, target_sr=16000)
output = pipe(
    audio,
    chunk_length_s=chunk_length_s-1,
    generate_kwargs=generate_kwargs,
    return_timestamps="word",
)
print(output)
