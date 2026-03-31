#!/usr/bin/env python3
import argparse
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate an image with Tongyi-MAI/Z-Image-Turbo from a prompt or prompt file."
    )
    parser.add_argument("--prompt", type=str, help="Prompt text")
    parser.add_argument("--prompt-file", type=Path, help="Path to prompt text file")
    parser.add_argument("--out", type=Path, default=Path("z_image_turbo.png"))
    parser.add_argument(
        "--model",
        type=str,
        default="Tongyi-MAI/Z-Image-Turbo",
        help="Hugging Face model id or local path",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for model loading",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=9,
        help="Number of denoising steps; 9 corresponds to 8 DiT forwards in the model card example",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="Guidance scale; keep 0.0 for Z-Image-Turbo per model card guidance",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        choices=["flash", "_flash_3"],
        help="Optional transformer attention backend",
    )
    parser.add_argument(
        "--compile-transformer",
        action="store_true",
        help="Compile the transformer for faster repeated inference",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable model CPU offload for lower VRAM usage",
    )
    return parser


def resolve_dtype(torch_module, dtype_name: str):
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "float32":
        return torch_module.float32
    return torch_module.bfloat16


def load_prompt(args):
    if args.prompt:
        return args.prompt.strip()
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    raise SystemExit("Provide --prompt or --prompt-file")


def main():
    args = build_parser().parse_args()

    import torch
    from diffusers import ZImagePipeline

    prompt = load_prompt(args)
    if not prompt:
        raise SystemExit("Prompt is empty")
    if args.height <= 0 or args.width <= 0:
        raise SystemExit("--height and --width must be greater than 0")
    if args.num_inference_steps <= 0:
        raise SystemExit("--num-inference-steps must be greater than 0")

    torch_dtype = resolve_dtype(torch, args.dtype)
    print(f"Loading model: {args.model}")
    print(f"device={args.device}, dtype={torch_dtype}")
    pipe = ZImagePipeline.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
    )

    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)

    if args.attention_backend:
        try:
            pipe.transformer.set_attention_backend(args.attention_backend)
            print(f"attention_backend={args.attention_backend}")
        except Exception as exc:
            print(
                f"Warning: unable to enable attention backend "
                f"{args.attention_backend!r}: {exc}"
            )
            print("Falling back to the default attention implementation.")

    if args.compile_transformer:
        pipe.transformer.compile()
        print("Compiled transformer")

    generator = torch.Generator(args.device).manual_seed(args.seed)
    image = pipe(
        prompt=prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    r"""
    Windows PowerShell:
    conda run --no-capture-output -n imagegen python -m scripts.image.z_image_turbo `
      --prompt-file "extensions\prompts\image_generator_prompt.txt" `
      --out "app\cache\images\z_image_turbo.png" `
      --device "cuda" `
      --dtype "bfloat16" `
      --height 512 `
      --width 512 `
      --num-inference-steps 9 `
      --attention-backend flash `
      --guidance-scale 0.0 `
      --seed 42
    """
    main()
