# Table of Contents

- [Quality benchmarks](#quality-benchmarks)
  - [Run evaluation](#run-evaluation)
- [Performance benchmarks](#performance-benchmarks)

# Quality benchmarks

For evaluation, we used datasets from [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard#evaluate-a-model).
For all evaluations we used following settings:

- **Metrics**: Word Error Rate (WER)
- **Text normalization**: Standard Whisper normalizer (lowercase, remove punctuation for WER)
- **Decoding**: `num_beams=1` with `do_sample=False`

**Huggingface Open-ASR-Leaderboard (English)**

| Dataset | TheWhisper | openai/whisper-large-v3-turbo | nvidia/parakeet-tdt-0.6b-v3 | ibm-granite/granite-speech-3.3-2b |
|---------|------------|----------------------------|--------------------------|--------------------------------|
| librispeech_clean_test | 1.73 | 2.1 | 1.92 | 1.53 |
| librispeech_other_test | 3.69 | 4.24 | 3.59 | 3.26 |
| spgispeech_test | 1.89 | 2.97 | 3.98 | 3.87 |
| tedlium_test | 3.34 | 3.57 | 2.8 | 3.57 |
| voxpopuli_test | 6.52 | 11.87 | 6.09 | 5.93 |
| gigaspeech_test | 9.58 | 10.14 | 9.57 | 10.69 |
| earnings22_test | 11.01 | 11.63 | 11.19 | 10.25 |
| ami_test | 9.52 | 16.13 | 11.39 | 8.9 |
| **Mean** | **5.91** | 7.83 | 6.32 | 6.00 |

**Multilingual Results**

The table below presents mean WER values for each language, averaged across three benchmark datasets: FLEURS [8], MLS [9], and Common Voice 23.

| Language | TheWhisper | openai/whisper-large-v3-turbo | nvidia/parakeet-tdt-0.6b-v3 | nvidia/canary-1b-v2 |
|----------|------------|-------------------------------|-----------------------------|---------------------|
| German | 4.15 | 4.91 | 5.04 | 4.96 |
| French | 5.08 | 7.97 | 5.39 | 4.86 |
| Italian | 4.50 | 6.40 | 5.59 | 5.66 |
| Spanish | 3.14 | 3.94 | 3.75 | 3.22 |
| Portuguese | 4.07 | 5.97 | 5.41 | 6.23 |
| Indonesian | 5.75 | 6.98 | - | - |
| Russian | 5.55 | 4.42 | 5.51 | - |
| Arabic | 9.31 | 10.57 | - | - |
| Hindi | 9.06 | 19.25 | - | - |
| English | 4.66 | 4.8 | 4.85 | 4.7 |

**Multilingual Open-ASR-Leaderboard**

| Model | Mean WER |
|-------|----------|
| **TheWhisper** | **4.30** |
| microsoft/Phi-4-multimodal-instruct | 4.60 |
| nvidia/canary-1b-v2 | 4.89 |
| nvidia/parakeet-tdt-0.6b-v3 | 5.05 |
| openai/whisper-large-v3-turbo | 5.44 |

**Noisy Audio Evaluation:**

We evaluate robustness to background noise by testing across different Signal-to-Noise Ratios (SNR) using noise samples from the MUSAN dataset [6]:

| SNR Level (db) | TheWhisper | nvidia/parakeet-tdt-0.6b-v3 |
|-----------|-----------------|------------------------------|
| Clean | 5.91 | 6.34 |
| 10 | 6.99  | 7.12 |
| 5 | 8.20   | 8.23 |
| 0 | 11.10 | 11.66 |

## Run evaluation

To reproduce metrics on open asr (english and multilingual) run following commands:

```bash
pip install -r benchmark/requirements.txt
```

```bash
python run_evaluation.py \
    --model_name TheStageAI/thewhisper-large-v3-turbo \
    --mode XL \
    --task open_asr \
    --batch_size 64 \
```

Here ``--mode XL`` means that fp16 model engines will be used. Use ``--mode S`` for evaluation of int8 quantized model version.

Set ``--task multilingual_open_asr`` for multilingual evaluation.

---

# Performance benchmarks

Below we will use the following denotions for `TheWhisper` model variants:
- `S` refers to the quantized version of the model
- `XL` indicates the fp16 model accelerated using `qlip.compiler` developed by TheStageAI

Measurements:
- TTFT (Time To First Token, in seconds): The latency measured from start of inference to when the first token is produced.
- RTFx (Real-Time Factor): The ratio of audio transcription time to audio duration. We used 10 min audio as input.

## thewhisper-large-v3-turbo

### NVIDIA L40S

**Batch Size: 1**

| Model         | TTFT   | RTFx |
|:--------------|-------:|-----:|
| S             | 0.0119  | 149.08       |
| XL            | 0.0115  | 152.07       |
| torch_compile | 0.0464 | 81.15        |
| faster_whisper| 2.2240 | 118.18       |
| whisperx      | 4.7130 | 132.42       |

**Batch Size: 32**

| Model         | TTFT   | RTFx |
|:--------------|-------:|-----:|
| S             | 0.3059  | 516.90       |
| XL            | 0.3101  | 518.06       |
| torch_compile | 1.508   | 345.25       |
| faster_whisper| 2.2234  | 174.27       |
| whisperx      | 2.5988  | 254.64       |


---

### NVIDIA H100 80GB HBM3

**Batch Size: 1**

| Model         | TTFT   | RTFx |
|:--------------|-------:|-----:|
| S             | 0.0098  | 161.45       |
| XL            | 0.0093  | 164.61       |
| torch_compile | 0.0303  | 108.20       |
| faster_whisper| 1.81685 | 135.92       |
| whisperx      | 4.09490 | 151.23       |

**Batch Size: 64**

| Model         | TTFT   | RTFx |
|:--------------|-------:|-----:|
| S             | 0.3175  | 2016.18       |
| XL            | 0.3206  | 1975.49       |
| torch_compile | 1.0602  |  637.36 |
| faster_whisper| 2.187   |  200.09 |


---

### NVIDIA GeForce RTX 4090

**Batch Size: 1**

| Model         | TTFT   | RTFx |
|:--------------|-------:|-----:|
| S             | 0.0123  | 159.85       |
| XL            | 0.0122  | 156.45       |
| torch_compile | 0.0566 | 69.61        |

**Batch Size: 24**

| Model         | TTFT   | RTFx |
|:--------------|-------:|-----:|
| S             | 0.2341  | 917.57       |
| XL            | 0.2237  | 925.53       |
| torch_compile | 1.3964  | 342.71       |

---

### NVIDIA GeForce RTX 5090

**Batch Size: 1**

| Model         | TTFT  | RTFx |
|:--------------|------:|-------------:|
| S             | 0.0213 | 208.22       |
| XL            | 0.0231 | 216.88       |
| torch_compile | 0.0427 | 193.85       |
| faster_whisper| 1.3008 | 126.30       |


**Batch Size: 32**

| Model         | TTFT   |    RTFx |
|:--------------|-------:|--------:|
| S             | 0.2135  | 1500.83 |
| XL            | 0.3791  | 1461.42 |
| torch_compile | 1.0602  |  637.36 |
| faster_whisper| 2.187   |  200.09 |



---

### Jetson-Thor

**Batch Size: 1**

| Model         | TTFT  | RTFx |
|:--------------|------:|-------------:|
| S             | 0.057  | 39.19        |
| XL            | 0.094  | 40.50        |
| torch_compile | 0.122  | 34.47        |

**Batch Size: 32**

| Model         | TTFT  |   RTFx |
|:--------------|------:|-------:|
| S             | 1.497  | 223.61 |
| XL            | 1.511  | 208.56 |
| torch_compile | 3.662  | 180.26 |
