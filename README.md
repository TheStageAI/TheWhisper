# TheWhisper: High-Performance Speech-to-Text

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face%20Weights-yellow)](https://huggingface.co/TheStageAI/thewhisper-large-v3-turbo/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU-green.svg)](#usage-deployment)
[![Apple Silicon](https://img.shields.io/badge/Apple-Silicon-black.svg)](#usage-deployment)

<img width="1500" height="440" alt="the whisper (6)" src="https://github.com/user-attachments/assets/a86c98a7-c587-40cb-9ed3-d3b5ba5e76f2" />

## 🚀 Overview

This repository aims to share and develop the most efficient speech-to-text and text-to-speech inference solution -with a strong focus on self-hosting, cloud hosting, and on-device inference across multiple devices. 

For the first release this repository provides **open-source transcription models** with **streaming inference support** and:
- Hugging Face open weights
- High-performance TheStage AI inference engines (NVIDIA GPU)
- CoreML engines for macOS / Apple Silicon with the lowest in the world power consumption for MacOS
- Local RestAPI with frontend examples using ReactJS and Electron [see for details](electron_app/README.md)

It is optimized for **low-latency**, **low power usage**, and **scalable** streaming transcription. Ideal for real-time captioning, live meetings, voice interfaces, and edge deployments.

<details>
  <summary><strong>📖 Table of Contents</strong></summary>

- [🚀 Overview](#-overview)
- [✨ Features](#-features)
- [⚡ Quick Start](#-quick-start)
- [🛠️ Support Matrix](#-support-matrix-and-system-requirements)
- [💡 Usage](#usage-deployment)
- [🖥️ Build On-Device Desktop Application for Apple](#-build-on-device-desktop-application-for-apple)
- [📊 Quality Benchmarks](#-quality-benchmarks)
  - [🍏 Apple Silicon Benchmarks](apple_benchmarks.md)
  - [⚡ NVIDIA GPU Benchmarks](nvidia_benchmarks.md)
- [🏢 Enterprise License Summary](#-enterprise-license-summary)
- [🙌 Acknowledgements](#-acknowledgements)

</details>

---

## ✨ Features

- Open weights fine-tuned versions of Whisper models
- Fine-tuned models support inference with 10s, 15s, 20s and 30s
- CoreML engines for macOS and Apple Silicon, ~2W of power consumption, ~2GB RAM usage
- Optimized engines for NVIDIA GPUs through TheStage AI [ElasticModels](https://docs.thestage.ai/elastic_models/docs/source/index.html) (free for small orgs)
- Streaming implementation (NVIDIA + macOS)
- Benchmarks: latency, memory, power, and ASR accuracy (OpenASR)
- Simple Python API, Examples of deployment for MacOS desktop app with Electron and ReactJS

<img width="1547" height="877" alt="apple m2 whisper (4)" src="https://github.com/user-attachments/assets/9404cdc0-b120-4ba1-9c65-4d42089ba623" />
<img width="1547" height="877" alt="nvidia l40s (2)" src="https://github.com/user-attachments/assets/7c318bb6-cbd6-42ce-b42f-096cd7a1070c" />

---


## 📦 Quick start

### Clone the repository
```bash
git clone https://github.com/TheStageAI/TheWhisper.git
cd TheWhisper
```
### Install for Apple
```bash
pip install .[apple]
```

### Install for Nvidia
```bash
pip install .[nvidia]
```

### Install for Nvidia with TheStage AI optmized engines
```bash
pip install .[nvidia] --extra-index-url https://thestage.jfrog.io/artifactory/api/pypi/pypi-thestage-ai-production/simple
pip install thestage
# additional dependencies
pip install flash_attn==2.8.2 --no-build-isolation
```

Then generate access token on [TheStage AI Platform](https://app.thestage.ai) in your profile and execute the following command:
```bash
thestage config set --api-token <YOUR_API_TOKEN>
```
-----

## 🏗️ Support Matrix and System Requirements

| **Feature** | **whisper-large-v3 (Nvidia)** | **whisper-large-v3 (Apple)** | **whisper-large-v3-turbo (Nvidia)** | **whisper-large-v3-turbo (Apple)** |
| --- | --- | --- | --- | --- |
| Streaming | ❌ | ✅ | ❌ | ✅ |
| Accelerated | ✅ | ✅ | ✅ | ✅ |
| Word Timestamps | ❌ | ✅ | ❌ | ✅ |
| Multilingual | ✅ | ✅ | ✅ | ✅ |
| 10s Chunk Mode | ✅ | ✅ | ✅ | ✅ |
| 15s Chunk Mode | ✅ | ✅ | ✅ | ✅ |
| 20s Chunk Mode | ✅ | ✅ | ✅ | ✅ |
| 30s Chunk Mode | ✅ | ✅ | ✅ | ✅ |

### Nvidia GPU Requirements

- **Supported GPUs:** RTX 4090, L40s
- **Operating System:** Ubuntu 20.04+
- **Minimum RAM:** 2.5 GB (5 GB recommended for large-v3 model)
- **CUDA Version:** 11.8 or higher
- **Driver Version:** 520.0 or higher
- **Python version**: 3.10-3.12

### Apple Silicon Requirements

- **Supported Chipsets:** M1, M1 Pro, M1 Max, M1 Ultra, M2, M2 Pro, M2 Max, M2 Ultra, M3, M3 Pro, M3 Max, M4, M4 Pro, M4 Max
- **Operating System:** macOS 15.0 (Ventura) or later, iOS 18.0 or later
- **Minimum RAM:** 2 GB (4 GB recommended for large-v3 model)
- **Python version**: 3.10-3.12

---

## ▶️ Usage / Deployment

### Apple Usage

```python
import torch
from thestage_speechkit.apple import ASRPipeline

model = ASRPipeline(
    model='TheStageAI/thewhisper-large-v3-turbo',
    # optimized model with ANNA
    model_size='S'
    chunk_length_s=10
)

# inference
result = model(
    "path_to_your_audio.wav", 
    max_batch_size=32,
    return_timestamps="word"
)

print(result["text"])
```

### Apple Usage with Streaming

```python
from thestage_speechkit.apple import StreamingPipeline
from thestage_speechkit.streaming import MicStream, FileStream, StdoutStream

streaming_pipe = StreamingPipeline(
    model='TheStageAI/thewhisper-large-v3-turbo',
    # Optimized model by ANNA
    model_size='S',
    # Window length
    chunk_length_s=10,
    platform='apple'
)

# set stride in miliseconds
mic_stream = MicStream(step_size_s=0.5)
output_stream = StdoutStream()

while True:
    chunk = mic_stream.next_chunk()
    if chunk:
        approved_text, assumption = streaming_pipe(chunk)
        output_stream.rewrite(approved_text, assumption)
    else:
        break
```

### Nvidia Usage (HuggingFace Transfomers)

```python
import torch
from thestage_speechkit.nvidia import ASRPipeline

model = ASRPipeline(
    model='TheStageAI/thewhisper-large-v3-turbo',
    # allowed: 10s, 15s, 20s, 30s
    chunk_length_s=10,
    # optimized TheStage AI engines
    device='cuda'
)

# inference
result = model(
    audio="path_to_your_audio.wav", 
    max_batch_size=32
)

print(result["text"])
```

### Nvidia Usage (TheStage AI engines)

```python
import torch
from thestage_speechkit.nvidia import ASRPipeline

model = ASRPipeline(
    model='TheStageAI/thewhisper-large-v3-turbo',
    # allowed: 10s, 15s, 20s, 30s
    chunk_length_s=10,
    # optimized TheStage AI engines
    mode='S',
    device='cuda'
)

# inference
result = model(
    "path_to_your_audio.wav", 
    max_batch_size=32
)

print(result["text"])
```
-----

## 💻 Build On-Device Desktop Application for Apple

You can build a macOS desktop app with real-time transcription. Find a simple ReactJS application here: **Link to React Frontend**
You can also download our app built using this backend here: [TheNotes for macOS](https://cdn.thestage.ai/production/cms_file_upload/1761693601-8ef0605f-a2e0-4bef-97c1-b61452e4f7dc/The%20Notes%20Package%20Oct%2028%202025.pkg)

-----

## 📊 Quality Benchmarks

TheWhisper is a fine-tuned Whisper model that can process audio chunks of any size up to 30 seconds. Unlike the original Whisper models, it doesn't require padding audio with silence to reach 30 seconds. We conducted quality benchmarking across different chunk sizes: 10, 15, 20, and 30 seconds. For quality benchmarks, we used the multilingual benchmarks [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard#evaluate-a-model).

<img width="1547" height="531" alt="vanilla whisper (1)" src="https://github.com/user-attachments/assets/f0c86e58-d834-4ac7-a06b-df3a7ae3e9e9" />
<img width="1547" height="458" alt="TheStage AI Whisper (1)" src="https://github.com/user-attachments/assets/17fb45a3-b33d-4c83-b843-69b0f0aa3f65" />


### 10s chunks

| Model | Mean WER |
|-------|-----------------|
| openai/whisper-large-v3-turbo | 7.81 |
| openai/whisper-large-v3 | 7.45 |
| thewhisper-large-v3-turbo | 7.88 |
| thewhisper-large-v3 | 7.8 |


### 15s chunks

| Model | Mean WER |
|-------|-----------------|
| openai/whisper-large-v3-turbo | 7.61 |
| openai/whisper-large-v3 | 7.22 |
| thewhisper-large-v3-turbo | 7.45 |
| thewhisper-large-v3 | 7.34 |

### 20s chunks

| Model | Mean WER |
|-------|-----------------|
| openai/whisper-large-v3-turbo | 7.63 |
| openai/whisper-large-v3 | 7.29 |
| thewhisper-large-v3-turbo | 7.47 |
| thewhisper-large-v3 | 7.31 |

### 30s chunks

| Model | Mean WER |
|-------|-----------------|
| openai/whisper-large-v3-turbo | 7.61 |
| openai/whisper-large-v3 | 7.32 |
| thewhisper-large-v3-turbo | 7.45 |
| thewhisper-large-v3 | 7.28 |

---

## 🏢 Enterprise License Summary

| Platform                 | Engine Type               | Status     | License                                 |
|--------------------------|---------------------------|------------|-----------------------------------------|
| NVIDIA GPUs (CUDA)       | TheStage AI (Optimized) | ✅ Stable  | Free ≤ 4 GPUs/year for small orgs       |
| NVIDIA GPUs (CUDA)       | Pytorch HF Transformers | ✅ Stable  | Free                                    |
| macOS / Apple Silicon    | CoreML Engine + MLX     | ✅ Stable  | Free                                    |

----

## 🙌 Acknowledgements


- **Silero VAD**: Used for voice activity detection in `thestage_speechkit/vad.py`. See https://github.com/snakers4/silero-vad.
- **OpenAI Whisper**: Original Whisper model and pretrained checkpoints. See https://github.com/openai/whisper.
- **Hugging Face Transformers**: Model, tokenizer, and inference utilities. See https://github.com/huggingface/transformers.
- **MLX community**: MLX Whisper implementation for Apple Silicon. See https://github.com/ml-explore/mlx-examples/tree/main/whisper.
