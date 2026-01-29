import numpy as np
from typing import Dict, Any
import sys
import torch
import transformers
import argparse
from transformers import WhisperForConditionalGeneration as HFWhisperForConditionalGeneration
from transformers import WhisperProcessor, pipeline

from data_utils import (
    open_asr_en_tasks, 
    open_asr_multilingual_tasks,
    DatasetConfig,
)
from eval_utils import evaluate_whisper

from elastic_models.transformers import WhisperForConditionalGeneration

import os
import json
from time import time as _now


def get_generator(args):
    dtype = torch.float16

    if args.mode == 'eager':
        chunk_length = 30
        model = HFWhisperForConditionalGeneration.from_pretrained(
            args.model_name, 
            cache_dir=args.cache_dir,
            torch_dtype=dtype,
        ).to("cuda")
        model.generation_config.forced_decoder_ids = None
    else:
        chunk_length = 20
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_name, 
            cache_dir=args.cache_dir,
            chunk_length=chunk_length,
            torch_dtype=dtype,
            mode=args.mode
        ).to("cuda")
        model.generation_config.forced_decoder_ids = None
        model.generation_config.cache_implementation = "flexi-static"

    if args.mode != "eager":
        processor = WhisperProcessor.from_pretrained(
            args.model_name, cache_dir=args.cache_dir, chunk_length=args.chunk_length,
            tokenizer=AutoTokenizer.from_pretrained(
                args.model_name, cache_dir=args.cache_dir, use_fast=True
            )
        )
    else:
        processor = WhisperProcessor.from_pretrained(
            args.model_name, cache_dir=args.cache_dir,
            tokenizer=AutoTokenizer.from_pretrained(
                args.model_name, cache_dir=args.cache_dir, use_fast=True
            )
        )

    generator = pipeline(
        "automatic-speech-recognition", 
        model=model, 
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device="cuda",
        batch_size=args.batch_size,
        chunk_length_s=chunk_length
    )

    return generator


def get_tasks(args):
    tasks = []
    if args.task == "open_asr":
        tasks = open_asr_en_tasks(
            min_duration_s=args.min_audio_len,
            max_duration_s=args.max_audio_len,
            split="test",
            cache_dir=args.cache_dir,
        )
    elif args.task == "multilingual_open_asr":
        tasks = open_asr_multilingual_tasks(
            min_duration_s=args.min_audio_len,
            max_duration_s=args.max_audio_len,
            split="test",
            cache_dir=args.cache_dir,
        )
    return tasks


def main(args):
    generate_kwargs = {
        "num_beams": 1, 
        "disable_compile": True, 
        'task': 'transcribe', 
        'do_sample': False, 
        'max_new_tokens': 256,
    }

    if args.mode != "eager":
        generate_kwargs["cache_implementation"] = "flexi-static"

    tasks = get_tasks(args)
    asr_generator = get_generator(args)


    def transcribe_fn(audio, generate_kwargs):
        return asr_generator(
            audio, generate_kwargs=generate_kwargs, chunk_length_s=20
        )

    save_path = f"{args.output_dir}/eval_results.json"

    results = evaluate_whisper(
        asr_generator=transcribe_fn, 
        generate_kwargs=generate_kwargs, 
        benchmark_tasks=tasks,
        batch_size=args.batch_size,
        save_path=save_path, 
    )

    print("\n" + "="*120)
    print(f"{'Task':<30} {'WER':<8} {'CER':<8} {'Duration (h)':<15} {'RTFx':<10}")
    print("="*120)
    for task_name, metrics in results.items():
        wer = metrics.get('wer', 0.0) * 100 if metrics.get('wer') is not None else 0.0
        cer = metrics.get('cer', 0.0) * 100 if metrics.get('cer') is not None else 0.0
        duration = metrics.get('dataset_duration_hours', 0.0)
        rtfx = metrics.get('rtfx', 0.0)
        print(f"{task_name:<30} {wer:<8.2f} {cer:<8.2f} {duration:<15.2f} {rtfx:<10.2f}")
    print("="*120 + "\n")


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="TheStageAI/thewhisper-large-v3-turbo")
    args.add_argument("--cache_dir", type=str, default="./huggingface_cache")
    args.add_argument("--min_audio_len", type=float, default=None)
    args.add_argument("--max_audio_len", type=float, default=None)
    args.add_argument("--batch_size", type=int, default=64)
    args.add_argument("--output_dir", type=str, default='./results', help="Output directory")
    args.add_argument("--mode", type=str, default="XL", choices=["eager", "S", "XL"])
    args.add_argument(
        "--task", 
        type=str, 
        default="open_asr", 
        choices=["open_asr", "multilingual_open_asr"], 
        help="Task to evaluate"
    )
    args = args.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
