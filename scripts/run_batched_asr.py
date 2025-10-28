from thestage_speechkit.nvidia import BatchedASRPipeline

chunk_length_s = 10
pipe = BatchedASRPipeline(
    model='TheStageAI/thewhisper-large-v3-turbo',
    chunk_length_s=chunk_length_s,
    device='cuda',
)
generate_kwargs={
    'num_beams': 1,
    'do_sample': False,
    'use_cache': True,
}
output = pipe(
    ['example_speech.wav', 'example_speech.wav'],
    generate_kwargs=generate_kwargs,
    chunk_length_s=chunk_length_s,
)
print(output)
