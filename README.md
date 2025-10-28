# 🗣️ TheWhisper: High-Performance Speech-to-Text engines

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face%20Weights-yellow)](https://huggingface.co/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU-green.svg)]()
[![Apple Silicon](https://img.shields.io/badge/Apple-Silicon-black.svg)]()

<img width="1500" height="440" alt="the whisper (6)" src="https://github.com/user-attachments/assets/a86c98a7-c587-40cb-9ed3-d3b5ba5e76f2" />

## 🚀 Overview

This repository aims to share and develop the most efficient speech-to-text and text-to-speech inference solution -with a strong focus on self-hosting, cloud hosting, and on-device inference across multiple devices. 

For the first release this repository provides **open-source transcription models** with **streaming inference support** and:
- Hugging Face open weights
- High-performance TheStage AI inference engines (NVIDIA GPU)
- CoreML engines for macOS / Apple Silicon with the lowest in the world power consumption for MacOS
- Local RestAPI with frontend examples using ReactJS and Electron

It is optimized for **low-latency**, **low power usage**, and **scalable** streaming transcription. Ideal for real-time captioning, live meetings, voice interfaces, and edge deployments.

<details>
  <summary><strong>📖 Table of Contents</strong></summary>

- [🚀 Overview](#-overview)
- [✨ Features](#-features)
- [🏗️ Supported Platforms](#️-supported-platforms)
- [📦 Installation](#-installation)
- [⚡ Quick Start](#-quick-start)
- [📈 Benchmarks](#-benchmarks)
  - [🍏 Apple Silicon Benchmarks](apple_benchmarks.md)
  - [⚡ NVIDIA GPU Benchmarks](nvidia_benchmarks.md)
- [📜 License](#-license)
- [🏢 Enterprise License Summary](#-enterprise-license-summary)
- [🧪 Evaluation](#-evaluation)
- [🤝 Contributing](#-contributing)
- [🙌 Acknowledgements](#-acknowledgements)
- [📬 Contact](#-contact)

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
pip install .[nvidia]
pip install thestage
pip install thestage_elastic_models[nvidia] --extra-index-url https://thestage.jfrog.io/artifactory/api/pypi/pypi-thestage-ai-production/simple
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
    model='thestage/thewhisper-large-v3-trubo',
    # optimized model with ANNA
    model_size='S'
    model_chunk='10s',
    device='cuda',
    hf_token=""
)

# inference
result = model(
	audio="path_to_your_audio.wav", 
	max_batch_size=32,
	return_timestamps="segment"
)

print(result["text"])
```

### Apple Usage with Streaming

```python
from thestage_speechkit.apple import WhisperStreamingPipeline
from thestage_speechkit.streaming import MicStream, FileStream, StdoutStream

streaming_model = WhisperStreaming(
	model='thewhisper-large-v3-turbo',
	# Optimized model by ANNA
	model_size='S',
  # Window length
	chunk_size='10s'
	platform
)

# set stride in miliseconds
mic_stream = MicStream(stride=500)
output_stream = StdoutStream()

while True:
	chunk = mic_stream.next_chunk()
	approved_text, assumption = streaming_model(chunk)
	output_stream.rewrite(approved_text, assumption)
```

### Nvidia Usage (HuggingFace Transfomers)

```python
import torch
from thestage_speechkit.nvidia import ASRPipeline

model = ASRPipeline(
    model='thestage/thewhisper-large-v3-trubo',
    # allowed: 10s, 15s, 20s, 30s
    model_chunk='10s',
    # optimized TheStage AI engines
    device='cuda',
    hf_token=""
)

# inference
result = model(
	audio="path_to_your_audio.wav", 
	max_batch_size=32,
	return_timestamps="segment"
)

print(result["text"])
```

### Nvidia Usage (TheStage AI engines)

```python
import torch
from thestage_speechkit.nvidia import ASRPipeline

model = ASRPipeline(
    model='thestage/thewhisper-large-v3-trubo',
    # allowed: 10s, 15s, 20s, 30s
    model_chunk='10s',
    # optimized TheStage AI engines
    mode='S',
    device='cuda',
    hf_token=""
)

# inference
result = model(
	audio="path_to_your_audio.wav", 
	max_batch_size=32,
	return_timestamps="segment"
)

print(result["text"])
```
-----

## 💻 Build On-Device Desktop Application for Apple

You can build a macOS desktop app with real-time transcription. Find a simple ReactJS application here: **Link to React Frontend**
You can also download our app built using this backend here: **ADD LINK TO APP**


https://github.com/user-attachments/assets/093c1442-faa5-4bb5-9885-cffd1dda1aa2

-----

## 📊 Quality Benchmarks

TheWhisper is a fine-tuned Whisper model that can process audio chunks of any size up to 30 seconds. Unlike the original Whisper models, it doesn't require padding audio with silence to reach 30 seconds. We conducted quality benchmarking across different chunk sizes: 10, 15, 20, and 30 seconds. For quality benchmarks, we used the [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard#evaluate-a-model)

<img width="1547" height="531" alt="vanilla whisper (1)" src="https://github.com/user-attachments/assets/f0c86e58-d834-4ac7-a06b-df3a7ae3e9e9" />
<img width="1547" height="458" alt="TheStage AI Whisper (1)" src="https://github.com/user-attachments/assets/17fb45a3-b33d-4c83-b843-69b0f0aa3f65" />


### 10s chunks

| Model | voxpopuli_test | tedlium_test | spgispeech_test | librispeech_test.other | librispeech_test.clean | gigaspeech_test | earnings22_test | ami_test | Mean WER |
|-------|-----------------|--------------|-----------------|------------------------|------------------------|-----------------|------------------|----------|----------|
| openai/whisper-large-v3-turbo (original) | 10.01 | 3.69 | 2.97 | 4.36 | 2.39 | 10.12 | 12.47 | 16.49 | 7.81 |
| openai/whisper-large-v3-turbo (truncated features) | 8.19 | 4.73 | 4.68 | 7.74 | 3.03 | 12.20 | 16.09 | 24.58 | 10.16 |
| **the-whisper-large-v3-turbo** | 7.41 | 3.88 | 3.11 | 4.84 | 2.44 | 10.61 | 12.88 | 17.86 | 7.88 |


## 🏢 Enterprise License Summary

| Platform                 | Engine Type               | Status     | License                                 |
|--------------------------|---------------------------|------------|-----------------------------------------|
| NVIDIA GPUs (CUDA)       | TheStage AI (Optimized) | ✅ Stable  | Free ≤ 4 GPUs/year for small orgs       |
| NVIDIA GPUs (CUDA)       | Pytorch HF Transformers | ✅ Stable  | Free                                    |
| macOS / Apple Silicon    | CoreML Engine + MLX     | ✅ Stable  | Free                                    |

----

## 🙌 Acknowledgements


