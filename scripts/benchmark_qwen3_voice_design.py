#!/usr/bin/env python3
import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


TEXT_PRESETS = {
    "short": (
        "Welcome back. We tuned the system for a warmer, more cinematic voice today."
    ),
    "medium": (
        "Welcome back. We tuned the system for a warmer, more cinematic voice today. "
        "This benchmark measures how quickly Qwen3-TTS can design and render a voice "
        "for a moderately sized passage while keeping the tone expressive and stable."
    ),
    "long": (
        "Welcome back. We tuned the system for a warmer, more cinematic voice today. "
        "This benchmark measures how quickly Qwen3-TTS can design and render a voice "
        "for a moderately sized passage while keeping the tone expressive and stable. "
        "The goal is not only to hear the result, but to understand how throughput "
        "changes as the text grows and the instruction prompt becomes more detailed. "
        "A useful benchmark should reflect realistic narration tasks, so this sample "
        "adds enough structure, pacing shifts, and descriptive phrasing to make the "
        "model work harder than it would on a single short sentence."
    ),
}


INSTRUCT_PRESETS = {
    "short": (
        "A warm, confident female narrator."
    ),
    "medium": (
        "A warm, confident female narrator in her early thirties with polished studio "
        "presence, steady pacing, and a calm, cinematic tone."
    ),
    "long": (
        "A warm, confident female narrator in her early thirties with polished studio "
        "presence, steady pacing, low-key emotional intelligence, and a calm cinematic "
        "tone. The voice should feel intimate but authoritative, with clear diction, "
        "gentle breaths, restrained dynamics, and a subtle documentary quality that "
        "makes long-form listening feel smooth and immersive."
    ),
}


@dataclass
class BenchmarkCase:
    text_label: str
    instruct_label: str
    text: str
    instruct: str


def resolve_dtype(name: str, device_map: str):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if "cuda" in device_map and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def configure_torch_for_inference(device_map: str):
    if "cuda" not in device_map or not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except RuntimeError:
        pass


def build_cases(selected_texts: list[str], selected_instructs: list[str]) -> list[BenchmarkCase]:
    cases = []
    for text_label in selected_texts:
        for instruct_label in selected_instructs:
            cases.append(
                BenchmarkCase(
                    text_label=text_label,
                    instruct_label=instruct_label,
                    text=TEXT_PRESETS[text_label],
                    instruct=INSTRUCT_PRESETS[instruct_label],
                )
            )
    return cases


def audio_duration_seconds(wav) -> float:
    audio = np.asarray(wav, dtype="float32").reshape(-1)
    return float(audio.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        help="HF model id or local model path",
    )
    ap.add_argument(
        "--device-map",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Model device_map (auto, cpu, cuda:0, mps)",
    )
    ap.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype for from_pretrained",
    )
    ap.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa" if torch.cuda.is_available() else None,
        help="Optional attention backend (e.g. sdpa, flash_attention_2)",
    )
    ap.add_argument("--language", type=str, default="English")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup-runs", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument(
        "--text-lengths",
        nargs="+",
        default=["short", "medium", "long"],
        choices=sorted(TEXT_PRESETS),
    )
    ap.add_argument(
        "--instruct-lengths",
        nargs="+",
        default=["short", "medium", "long"],
        choices=sorted(INSTRUCT_PRESETS),
    )
    ap.add_argument(
        "--save-audio",
        action="store_true",
        help="Save generated wav files for the last repeat of each case",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("app/cache/benchmarks/qwen3_voice_design"),
    )
    args = ap.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs cannot be negative")

    configure_torch_for_inference(args.device_map)
    load_kwargs = {
        "device_map": args.device_map,
        "dtype": resolve_dtype(args.dtype, args.device_map),
    }
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation

    print(f"Loading model: {args.model}")
    print(f"device_map={args.device_map}, dtype={load_kwargs['dtype']}")
    if args.attn_implementation:
        print(f"attn_implementation={args.attn_implementation}")
    model = Qwen3TTSModel.from_pretrained(args.model, **load_kwargs)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark_results.csv"

    cases = build_cases(args.text_lengths, args.instruct_lengths)
    rows = []

    for case in cases:
        print(
            f"\nCase text={case.text_label} ({len(case.text)} chars), "
            f"instruct={case.instruct_label} ({len(case.instruct)} chars)"
        )

        for warmup_idx in range(args.warmup_runs):
            print(f"  warmup {warmup_idx + 1}/{args.warmup_runs}")
            with torch.inference_mode():
                model.generate_voice_design(
                    text=case.text,
                    language=args.language,
                    instruct=case.instruct,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        run_rtfs = []
        run_latencies = []
        run_audio_durations = []

        for repeat_idx in range(args.repeats):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                wavs, sample_rate = model.generate_voice_design(
                    text=case.text,
                    language=args.language,
                    instruct=case.instruct,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

            wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
            audio_secs = audio_duration_seconds(wav) / float(sample_rate)
            rtf = elapsed / audio_secs if audio_secs > 0 else float("inf")

            run_latencies.append(elapsed)
            run_audio_durations.append(audio_secs)
            run_rtfs.append(rtf)

            if args.save_audio and repeat_idx == args.repeats - 1:
                audio_path = out_dir / (
                    f"voice_design_{case.text_label}_{case.instruct_label}.wav"
                )
                sf.write(audio_path, wav, sample_rate)

            print(
                f"  run {repeat_idx + 1}/{args.repeats}: "
                f"{elapsed:.2f}s wall, {audio_secs:.2f}s audio, RTF={rtf:.3f}"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        row = {
            "text_length": case.text_label,
            "instruct_length": case.instruct_label,
            "text_chars": len(case.text),
            "instruct_chars": len(case.instruct),
            "repeats": args.repeats,
            "median_wall_seconds": round(statistics.median(run_latencies), 4),
            "median_audio_seconds": round(statistics.median(run_audio_durations), 4),
            "median_rtf": round(statistics.median(run_rtfs), 4),
            "min_rtf": round(min(run_rtfs), 4),
            "max_rtf": round(max(run_rtfs), 4),
            "device_map": args.device_map,
            "dtype": str(load_kwargs["dtype"]),
            "attn_implementation": args.attn_implementation or "",
            "do_sample": args.do_sample,
            "model": args.model,
        }
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV: {csv_path}")
    print("\nSummary (lower RTF is better):")
    for row in rows:
        print(
            f"  text={row['text_length']:<6} instruct={row['instruct_length']:<6} "
            f"median_rtf={row['median_rtf']:.3f} "
            f"median_wall={row['median_wall_seconds']:.2f}s "
            f"median_audio={row['median_audio_seconds']:.2f}s"
        )


if __name__ == "__main__":
    """
    Windows PowerShell:
    conda run --no-capture-output -n qwen3-tts-cuda python -u scripts/benchmark_qwen3_voice_design.py `
      --device-map "cuda:0" `
      --attn-implementation "sdpa" `
      --dtype "bfloat16" `
      --repeats 3 `
      --warmup-runs 1 `
      --save-audio
    """
    main()
