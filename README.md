# 🗣️ TheWhisper: High-Performance Voice AI engines

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face%20Weights-yellow)](https://huggingface.co/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU-green.svg)]()
[![Apple Silicon](https://img.shields.io/badge/Apple-Silicon-black.svg)]()

<img width="1500" height="440" alt="the whisper (6)" src="https://github.com/user-attachments/assets/a86c98a7-c587-40cb-9ed3-d3b5ba5e76f2" />

## 🚀 Overview

The goal of the repository is to share @TheStageAI
This repository provides **open-source transcription models** with **streaming inference support** and:
- Hugging Face open weights
- High-performance TheStage AI inference engines (NVIDIA GPU)
- CoreML engines for macOS / Apple Silicon with the lowest in the world power consumption for MacOS
- Local RestAPI with frontend examples using ReactJS and Electron

It is optimized for **low-latency**, **low power usage**, and **scalable** streaming transcription. <br>
Ideal for real-time captioning, live meetings, voice interfaces, and edge deployments.

<img width="1547" height="877" alt="apple m2 whisper" src="https://github.com/user-attachments/assets/f9a7ed1c-6c0a-4497-accd-f9adf57f6845" />
<img width="1547" height="877" alt="nvidia l40s (1) (1)" src="https://github.com/user-attachments/assets/680d4da7-85ff-48dc-9273-755a3be8c39c" />

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


---

## 🏗️ Supported Platforms

| Platform                 | Engine Type               | Status     | License                                 |
|--------------------------|---------------------------|------------|-----------------------------------------|
| NVIDIA GPUs (CUDA)       | TheStage AI (Optimized) | ✅ Stable  | Free ≤ 4 GPUs/year for small orgs       |
| NVIDIA GPUs (CUDA)       | Pytorch HF Transformers | ✅ Stable  | Free                                    |
| macOS / Apple Silicon    | CoreML Engine + MLX     | ✅ Stable  | Free                                    |

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


## Streaming transcription


