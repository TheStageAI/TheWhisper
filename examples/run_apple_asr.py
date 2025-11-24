from librosa import load, resample
from thestage_speechkit.apple import ASRPipeline

chunk_length_s = 15
pipe = ASRPipeline('TheStageAI/thewhisper-large-v3-turbo', chunk_length_s=chunk_length_s, model_size='S')
generate_kwargs={
    'max_new_tokens': 128,
    'num_beams': 1,
    'do_sample': False,
    'use_cache': True,
}
audio, sr = load('example_speech.wav')
audio = resample(audio, orig_sr=sr, target_sr=16000)
output = pipe(
    audio,
    chunk_length_s=chunk_length_s-1,
    generate_kwargs=generate_kwargs, 
    return_timestamps='word'
)
print(output)
