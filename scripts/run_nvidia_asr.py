from thestage_asr.nvidia import ASRPipeline

pipe = ASRPipeline(
    'openai/whisper-large-v3-turbo', chunk_length_s=10, model_size='S'
)
output = pipe('example_speech.wav', chunk_length_s=10)
print(output)
