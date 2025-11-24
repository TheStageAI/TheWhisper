from librosa import load, resample
from thestage_speechkit.nvidia import ASRPipeline

generate_kwargs={
    'num_beams': 1,
    'do_sample': False,
    'use_cache': True,
    'language': 'en',
}
chunk_length_s = 10
pipe = ASRPipeline(
    'TheStageAI/thewhisper-large-v3-turbo', 
    chunk_length_s=chunk_length_s, 
    model_size='S', 
    batch_size=32,
    device='cuda'
)
audio, sr = load('example_speech.wav')
audio = resample(audio, orig_sr=sr, target_sr=16000)
output = pipe(
    audio,
    generate_kwargs=generate_kwargs,
    chunk_length_s=chunk_length_s-1,
)
print(output)
