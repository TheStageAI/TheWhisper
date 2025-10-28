from thestage_asr.apple import ASRPipeline
from thestage_asr.apple.model import TheWhisperForConditionalGeneration

pipe = ASRPipeline('TheStageAI/thewhisper-large-v3-turbo', chunk_length_s=10)
print(pipe.feature_extractor)
# generate_kwargs={
#     'max_new_tokens': 256,
#     'num_beams': 1,
#     'do_sample': False,
#     'use_cache': True,
#     'language': 'en',
# }
print(pipe.model.config)
output = pipe(
    'sample_1.mp3', 
    chunk_length_s=10, 
    # generate_kwargs=generate_kwargs, 
    # return_timestamps='word'
)
print(output)
