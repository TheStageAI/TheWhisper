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
provider. `--coval_trim` sets the cut point:

| mode | cut point |
|---|---|
| `manifest` | offset from the Coval manifest (what the leaderboard uses) |
| `none` | no cut, full clip |
| `clean-twin` | offset of the same clip in `stt-wildasr-clean` |

The WildASR sets are the clean clips with an augmentation applied. Reverb and far-field add a
tail after the last word, noise-gap inserts silence between speech segments, clipping and phone
codec keep the length. So the end of speech in the clean clip is also the end of speech in the
augmented one; for noise-gap the offset is shifted by the added silence. `stt-v3` and
`stt-wildasr-accent` have no clean version and use their own offsets.

On `stt-wildasr-reverb` some manifest offsets are placed before the end of speech, so the last
words are cut off and counted as deletions.

## Results

TheWhisper, compiled XL engine, revision `355761f9`, WER %.

| dataset | `manifest` | `clean-twin` | `none` |
|---|---|---|---|
| stt-v3 | 3.04 | 3.04 | 3.52 |
| stt-wildasr-accent | 2.96 | 2.96 | 2.96 |
| stt-wildasr-clean | 4.16 | 4.16 | 4.30 |
| stt-wildasr-clipping | 5.79 | 5.86 | 5.93 |
| stt-wildasr-farfield | 5.67 | 5.67 | 5.63 |
| stt-wildasr-noisegap | 6.55 | 6.51 | 6.51 |
| stt-wildasr-phonecodec | 4.98 | 5.01 | 5.13 |
| **stt-wildasr-reverb** | **11.75** | **5.56** | **5.55** |
| mean | 5.61 | 4.85 | 4.94 |

WER is computed as in Coval: Whisper `EnglishTextNormalizer`, errors summed over the whole
dataset and divided by the total number of reference words.
