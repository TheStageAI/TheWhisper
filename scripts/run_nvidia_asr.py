from thestage_asr.nvidia import ASRPipeline

generate_kwargs={
    'num_beams': 1,
    'do_sample': False,
    'use_cache': True,
    'language': 'en',
}
pipe = ASRPipeline(
    'openai/whisper-large-v3-turbo', 
    chunk_length_s=10, model_size='S', device='cuda'
)
output = pipe('example_speech.wav', chunk_length_s=10, generate_kwargs=generate_kwargs)
print(output)
