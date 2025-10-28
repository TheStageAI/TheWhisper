from thestage_asr.nvidia import BatchedASRPipeline

pipe = BatchedASRPipeline(
    model='TheStageAI/thewhisper-large-v3-turbo',
    chunk_length_s=10,
    model_size='S',
    device='cuda',
)
generate_kwargs={
    'num_beams': 1,
    'do_sample': False,
    'use_cache': True,
    'language': 'en',
}
output = pipe(
    ['example_speech.wav', 'example_speech.wav'],
    generate_kwargs=generate_kwargs,
    chunk_length_s=10,
)
print(output)
