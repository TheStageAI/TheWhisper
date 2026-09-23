# Coval STT benchmark

Evaluation of TheWhisper on the [Coval STT benchmark](https://benchmarks.coval.ai/stt):
`stt-v3` and the seven `stt-wildasr-*` datasets.

## Install

The benchmark runs the compiled TheStage AI engines, so install TheWhisper with the engines first
(see [Install for Nvidia with TheStage AI optimized engines](../../README.md#install-for-nvidia-with-thestage-ai-optmized-engines)),
then the benchmark requirements. From the repository root:

```bash
pip install 'thestage-elastic-models[nvidia]==0.2.2.post0' --index-url https://thestage.jfrog.io/artifactory/api/pypi/pypi-thestage-ai-production/simple --extra-index-url https://pypi.nvidia.com --extra-index-url https://pypi.org/simple
pip install .[nvidia]
pip install thestage
pip install -r benchmark/requirements.txt

thestage config set -t <YOUR_API_TOKEN>
```

The engines are downloaded with a TheStage AI token, which you can generate in your profile on
[TheStage AI Platform](https://app.thestage.ai).

## Run

```bash
cd benchmark

python run_evaluation.py --task coval --coval_trim manifest \
    --pipeline vad --chunk_length 30 \
    --revision 355761f94651e8c9f3639a23d8eca6253818076b \
    --mode XL --batch_size 64
```

Manifests are taken from the Coval GitHub repo at a fixed commit. Audio is downloaded from the
Coval public bucket into memory (about 0.9 GB per run) and checked against the sha256 in the
manifest, so these are the same files Coval uses.

Always set `--revision`: without it `elastic_models` loads the latest tag of the model repo,
which is an older checkpoint than `main`.

## Truncation modes

Coval cuts each clip at `speech_end_offset_ms` from the manifest before sending it to a
provider. `--coval_trim` selects the audio:

| mode | audio |
|---|---|
| `manifest` | cut at the offset from the Coval manifest (what the leaderboard uses) |
| `none` | the same clip without the cut |

On `stt-wildasr-reverb` some manifest offsets are placed before the end of speech, so the last
words are cut off and counted as deletions.

## Results

TheWhisper, compiled XL engine, revision `355761f9`, WER %.

| dataset | `none` (full audio) | `manifest` (Coval cut) |
|---|---|---|
| stt-v3 | 3.52 | 3.04 |
| stt-wildasr-accent | 2.96 | 2.96 |
| stt-wildasr-clean | 4.30 | 4.16 |
| stt-wildasr-clipping | 5.93 | 5.79 |
| stt-wildasr-farfield | 5.63 | 5.67 |
| stt-wildasr-noisegap | 6.51 | 6.55 |
| stt-wildasr-phonecodec | 5.13 | 4.98 |
| **stt-wildasr-reverb** | **5.55** | **11.75** |
| all datasets, pooled | 4.51 | 4.88 |

WER is computed as in Coval: Whisper `EnglishTextNormalizer`, errors summed over the whole
dataset and divided by the total number of reference words. The last row pools all eight
datasets the same way, which is how Coval combines datasets.
