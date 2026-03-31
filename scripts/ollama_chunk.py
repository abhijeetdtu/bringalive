#!/usr/bin/env python3
import argparse
import json
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    index: int
    total: int
    text: str

    @property
    def output_name(self) -> str:
        return f"chunk_{self.index:04d}.txt"


class SentenceChunker:
    SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_chars: int):
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than 0")
        self.max_chars = max_chars

    def normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def split_sentences(self, text: str):
        normalized = self.normalize_text(text)
        if not normalized:
            return []
        parts = self.SENT_SPLIT_RE.split(normalized)
        return parts if parts else [normalized]

    def pack_sentences(self, sentences):
        chunks = []
        buf = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if not buf:
                if len(sentence) <= self.max_chars:
                    buf = sentence
                else:
                    chunks.extend(self._split_long_sentence(sentence))
            else:
                candidate = f"{buf} {sentence}"
                if len(candidate) <= self.max_chars:
                    buf = candidate
                else:
                    chunks.append(buf)
                    if len(sentence) <= self.max_chars:
                        buf = sentence
                    else:
                        chunks.extend(self._split_long_sentence(sentence))
                        buf = ""
        if buf:
            chunks.append(buf)
        return chunks

    def chunk_text(self, text: str):
        packed = self.pack_sentences(self.split_sentences(text))
        total = len(packed)
        return [
            Chunk(index=idx, total=total, text=chunk_text)
            for idx, chunk_text in enumerate(packed, start=1)
        ]

    def _split_long_sentence(self, sentence: str):
        return [
            sentence[i: i + self.max_chars]
            for i in range(0, len(sentence), self.max_chars)
        ]


class PromptTemplate:
    def __init__(self, template_text: str):
        self.template_text = template_text

    @classmethod
    def from_file(cls, path: Path):
        return cls(path.read_text(encoding="utf-8"))

    def render(self, chunk: Chunk) -> str:
        if "{{chunk}}" in self.template_text:
            prompt = self.template_text.replace("{{chunk}}", chunk.text)
        else:
            prompt = f"{self.template_text.rstrip()}\n\n{chunk.text}"

        prompt = prompt.replace("{{chunk_index}}", str(chunk.index))
        prompt = prompt.replace("{{chunk_count}}", str(chunk.total))
        return prompt


class OllamaClient:
    def __init__(
        self,
        model: str,
        api_url: str,
        temperature: float | None = None,
        request_timeout: float = 10.0,
    ):
        self.model = model
        self.api_url = api_url
        self.temperature = temperature
        self.request_timeout = request_timeout

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if self.temperature is not None:
            payload["options"] = {"temperature": self.temperature}

        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout) as response:
            body = response.read().decode("utf-8")

        data = json.loads(body)
        if "response" not in data:
            raise RuntimeError(f"Unexpected Ollama response: {data}")
        return data["response"]


class ChunkCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def default_for(cls, text_file: Path, model: str):
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "model"
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", text_file.stem).strip("_") or "input"
        return cls(Path("app/cache/ollama") / safe_model / safe_stem)

    def output_path(self, chunk: Chunk) -> Path:
        return self.cache_dir / chunk.output_name

    def exists(self, chunk: Chunk) -> bool:
        return self.output_path(chunk).exists()

    def save_output(self, chunk: Chunk, text: str):
        self.output_path(chunk).write_text(text, encoding="utf-8")

    def write_manifest(self, manifest: dict):
        (self.cache_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )


class OllamaChunkProcessor:
    def __init__(
        self,
        text_file: Path,
        prompt_file: Path,
        chunker: SentenceChunker,
        prompt_template: PromptTemplate,
        client: OllamaClient,
        cache: ChunkCache,
        resume: bool = False,
    ):
        self.text_file = text_file
        self.prompt_file = prompt_file
        self.chunker = chunker
        self.prompt_template = prompt_template
        self.client = client
        self.cache = cache
        self.resume = resume

    def run(self):
        text = self.text_file.read_text(encoding="utf-8")
        chunks = self.chunker.chunk_text(text)
        if not chunks:
            raise SystemExit("Input text is empty after normalization")

        self.cache.write_manifest(
            {
                "text_file": str(self.text_file),
                "prompt_file": str(self.prompt_file),
                "model": self.client.model,
                "api_url": self.client.api_url,
                "chunk_count": len(chunks),
                "max_chars": self.chunker.max_chars,
            }
        )

        for chunk in chunks:
            if self.resume and self.cache.exists(chunk):
                print(f"[{chunk.index}/{chunk.total}] exists, skipping")
                continue

            prompt = self.prompt_template.render(chunk)
            print(f"[{chunk.index}/{chunk.total}] sending {len(chunk.text)} chars")
            try:
                result = self.client.generate(prompt)
            except KeyboardInterrupt:
                print(
                    f"\nInterrupted during chunk {chunk.index}. "
                    "Saved chunk files remain in cache."
                )
                raise SystemExit(130)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise SystemExit(
                    f"Ollama HTTP error for chunk {chunk.index}: {exc.code} {detail}"
                ) from exc
            except TimeoutError as exc:
                raise SystemExit(
                    f"Ollama request timed out for chunk {chunk.index} after "
                    f"{self.client.request_timeout} seconds"
                ) from exc
            except socket.timeout as exc:
                raise SystemExit(
                    f"Ollama request timed out for chunk {chunk.index} after "
                    f"{self.client.request_timeout} seconds"
                ) from exc
            except urllib.error.URLError as exc:
                raise SystemExit(
                    f"Unable to reach Ollama at {self.client.api_url} "
                    f"for chunk {chunk.index}: {exc}"
                ) from exc

            self.cache.save_output(chunk, result)
            print(f"[{chunk.index}/{chunk.total}] saved {self.cache.output_path(chunk)}")

        print(f"Done. Cached {len(chunks)} chunk responses in {self.cache.cache_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Chunk a text file, send each chunk to Ollama, and cache each response."
    )
    parser.add_argument("--text-file", type=Path, required=True, help="Path to the input text file")
    parser.add_argument("--prompt-file", type=Path, required=True, help="Path to the prompt template file")
    parser.add_argument("--model", type=str, required=True, help="Ollama model name, e.g. llama3.1")
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:11434/api/generate",
        help="Ollama generate API URL",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for cached chunk outputs; defaults to app/cache/ollama/<model>/<input-stem>",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=4000,
        help="Maximum characters per text chunk",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional Ollama temperature override",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=10.0,
        help="Socket timeout in seconds for each Ollama request",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip chunk outputs that already exist",
    )
    return parser


def build_processor(args):
    if not args.text_file.exists():
        raise SystemExit(f"Text file not found: {args.text_file}")
    if not args.prompt_file.exists():
        raise SystemExit(f"Prompt file not found: {args.prompt_file}")

    chunker = SentenceChunker(max_chars=args.max_chars)
    prompt_template = PromptTemplate.from_file(args.prompt_file)
    client = OllamaClient(
        model=args.model,
        api_url=args.api_url,
        temperature=args.temperature,
        request_timeout=args.request_timeout,
    )
    cache = (
        ChunkCache(args.cache_dir)
        if args.cache_dir is not None
        else ChunkCache.default_for(args.text_file, args.model)
    )
    return OllamaChunkProcessor(
        text_file=args.text_file,
        prompt_file=args.prompt_file,
        chunker=chunker,
        prompt_template=prompt_template,
        client=client,
        cache=cache,
        resume=args.resume,
    )


def main():
    args = build_parser().parse_args()
    try:
        build_processor(args).run()
    except KeyboardInterrupt:
        raise SystemExit("\nInterrupted by user")


if __name__ == "__main__":
    r"""
    If the prompt file contains `{{chunk}}`, the chunk text will be inserted there.
    Otherwise, the chunk text is appended after the prompt file contents.

    Optional placeholders:
    - {{chunk}}
    - {{chunk_index}}
    - {{chunk_count}}

    Example:
    python -m scripts.ollama_chunk \
        --text-file "bringalive/documents/Blue Highways.txt" \
        --prompt-file "extensions/openwebui_chunk_reader/prompt.txt" \
        --model "llama3.1" \
        --cache-dir "app/cache/ollama/blue_highways" \
        --request-timeout 10 \
        --max-chars 4000 \
        --resume

    Windows PowerShell:
    python -m scripts.ollama_chunk `
        --text-file "bringalive\documents\long_test.txt" `
        --prompt-file "extensions\openwebui_chunk_reader\prompt.txt" `
        --model "ministral-3:14b" `
        --cache-dir "app\cache\ollama\long_test" `
        --request-timeout 1000 `
        --max-chars 1000 `
        --resume
    """
    main()
