from thestage_asr.nvidia import ASRPipeline

pipe = ASRPipeline(
    'openai/whisper-large-v3-turbo', chunk_length_s=10
)
output = pipe('sample_1.mp3', chunk_length_s=10)
print(output)
