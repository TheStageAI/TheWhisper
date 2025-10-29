from librosa import load, resample
from thestage_speechkit.apple import ASRPipeline

pipe = ASRPipeline('TheStageAI/thewhisper-large-v3-turbo', chunk_length_s=10, model_size='S')
generate_kwargs={
    'max_new_tokens': 128,
    'num_beams': 1,
    'do_sample': False,
    'use_cache': True,
    'language': 'en',
}
audio, sr = load('example_speech.wav')
audio = resample(audio, orig_sr=sr, target_sr=16000)
output = pipe(
    audio,
    chunk_length_s=10, 
    generate_kwargs=generate_kwargs, 
    return_timestamps='word'
)
print(output)
