"""
title: Chunk Reader Filter
author: @codex
version: 0.4.0
required_open_webui_version: 0.5.0
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class SessionState:
    source_name: str = "Untitled"
    source_text: str = ""
    chunks: list[str] = field(default_factory=list)
    current_index: int = 0
    chunk_size: int = 1800


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0, description="Lower values run first.")
        default_chunk_size: int = Field(
            default=1800,
            description="Default chunk size used when loading text.",
        )
        default_summary_bullets: int = Field(
            default=3,
            description="Default number of bullets requested from the model.",
        )
        max_preview_chars: int = Field(
            default=4000,
            description="Maximum chunk text sent to the underlying model.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.sessions: dict[str, SessionState] = {}

    async def inlet(
        self,
        body: dict,
        __files__=None,
        __task__: str | None = None,
        __event_emitter__=None,
        __user__: dict | None = None,
        __metadata__: dict | None = None,
    ) -> dict:
        if __task__ is not None:
            return body

        chat_id = self._chat_id(body, __metadata__)
        session = self.sessions.setdefault(
            chat_id, SessionState(chunk_size=self.valves.default_chunk_size)
        )

        message = self._latest_user_message(body).strip()
        file_text = self._extract_text_from_files(__files__ or body.get("files") or [])
        lower = message.lower() if message else ""

        if not message and not file_text:
            return body

        if lower in {"reset", "/reset", "clear"}:
            self.sessions[chat_id] = SessionState(
                chunk_size=self.valves.default_chunk_size
            )
            self._replace_latest_user_message(
                body,
                "Chunk reader state reset. Upload or paste a new document.",
            )
            await self._emit_status(__event_emitter__, "Reader state reset.", done=True)
            return body

        if lower in {"help", "/help", "commands"}:
            self._replace_latest_user_message(body, self._help_prompt())
            return body

        size = self._extract_chunk_size(message)
        if size is not None:
            session.chunk_size = size
            if session.source_text:
                session.chunks = self._chunk_text(session.source_text, size)
                session.current_index = min(session.current_index, len(session.chunks) - 1)
                prompt = self._chunk_display_prompt(
                    session,
                    intro=f"The chunk size was updated to {size} and the document was re-chunked.",
                )
                self._replace_latest_user_message(body, prompt)
                await self._emit_status(__event_emitter__, "Chunk size updated.", done=True)
                return body
            self._replace_latest_user_message(
                body,
                f"Chunk size set to {size}. Load a document.",
            )
            return body

        load_text = self._extract_load_text(message, file_text=file_text)
        if load_text is not None:
            self._load_document(session, load_text, source_name="Attached file" if file_text else None)
            self._replace_latest_user_message(body, self._chunk_only_message(session))
            await self._emit_status(__event_emitter__, "Document loaded.", done=True)
            return body

        if file_text and not session.source_text:
            self._load_document(session, file_text, source_name="Attached file")
            self._replace_latest_user_message(body, self._chunk_only_message(session))
            await self._emit_status(__event_emitter__, "Attached file loaded.", done=True)
            return body

        if not session.source_text and self._looks_like_document(message):
            self._load_document(session, message, source_name="Pasted text")
            self._replace_latest_user_message(body, self._chunk_only_message(session))
            await self._emit_status(__event_emitter__, "Pasted text loaded.", done=True)
            return body

        if lower in {"status", "/status"}:
            self._replace_latest_user_message(body, self._status_prompt(session))
            return body

        if lower in {"show", "current", "/show"}:
            self._replace_latest_user_message(body, self._chunk_only_message(session))
            return body

        if lower in {"next", "n", "/next"}:
            self._move(session, 1)
            self._replace_latest_user_message(body, self._chunk_only_message(session))
            return body

        if lower in {"prev", "previous", "p", "/prev"}:
            self._move(session, -1)
            self._replace_latest_user_message(body, self._chunk_only_message(session))
            return body

        if lower in {"first", "/first"}:
            self._jump(session, 0)
            self._replace_latest_user_message(body, self._chunk_only_message(session))
            return body

        if lower in {"last", "/last"}:
            self._jump(session, len(session.chunks) - 1)
            self._replace_latest_user_message(body, self._chunk_only_message(session))
            return body

        if lower.startswith("restart ") or lower.startswith("start "):
            self._goto(session, message)
            self._replace_latest_user_message(body, self._chunk_only_message(session))
            return body

        if lower.startswith("goto ") or lower.startswith("go to "):
            self._goto(session, message)
            self._replace_latest_user_message(body, self._chunk_only_message(session))
            return body

        if self._is_summarize_request(lower):
            self._replace_latest_user_message(
                body,
                self._user_request_with_chunk(message, session),
            )
            await self._emit_status(__event_emitter__, "Passing current chunk to the model.", done=True)
            return body

        return body

    async def outlet(
        self,
        body: dict,
        __user__: dict | None = None,
        __metadata__: dict | None = None,
    ) -> dict:
        chat_id = self._chat_id(body, __metadata__)
        session = self.sessions.get(chat_id)
        if not session or not session.chunks:
            return body

        messages = body.get("messages") or []
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                return body
            header = self._chunk_header(session)
            footer = f"\n\n---\n{header}"
            if content.endswith(footer) or header in content.split("\n---\n")[-1]:
                return body
            message["content"] = f"{content}{footer}"
            return body

        return body

    def _chat_id(self, body: dict, metadata: dict | None) -> str:
        return (
            str(body.get("chat_id") or "")
            or str(body.get("conversation_id") or "")
            or str(body.get("metadata", {}).get("chat_id") or "")
            or str((metadata or {}).get("chat_id") or "")
            or "default"
        )

    def _latest_user_message(self, body: dict) -> str:
        messages = body.get("messages") or []
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                if parts:
                    return "\n".join(parts)
        content = body.get("content")
        return content if isinstance(content, str) else ""

    def _replace_latest_user_message(self, body: dict, new_content: str) -> None:
        messages = body.get("messages") or []
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            if isinstance(message.get("content"), str):
                message["content"] = new_content
                return
            if isinstance(message.get("content"), list):
                message["content"] = new_content
                return
        body["messages"] = (messages if messages else []) + [
            {"role": "user", "content": new_content}
        ]

    def _extract_load_text(
        self,
        message: str,
        file_text: str | None = None,
    ) -> str | None:
        lower = message.lower()
        for prefix in ("load\n", "load\r\n", "load: ", "load "):
            if lower.startswith(prefix):
                text = message[len(prefix):].strip()
                if text:
                    return text
        if lower in {"load file", "load attachment"} and file_text:
            return file_text
        return None

    def _extract_text_from_files(self, files: list[dict[str, Any]]) -> str | None:
        for file in files:
            if not isinstance(file, dict):
                continue
            candidates = [
                file.get("content"),
                file.get("text"),
                file.get("data"),
                file.get("file", {}).get("content") if isinstance(file.get("file"), dict) else None,
                file.get("file", {}).get("data") if isinstance(file.get("file"), dict) else None,
                file.get("data", {}).get("content") if isinstance(file.get("data"), dict) else None,
                file.get("data", {}).get("text") if isinstance(file.get("data"), dict) else None,
                file.get("meta", {}).get("content") if isinstance(file.get("meta"), dict) else None,
                file.get("meta", {}).get("text") if isinstance(file.get("meta"), dict) else None,
            ]
            for candidate in candidates:
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()

            for path_candidate in self._file_paths(file):
                text = self._read_text_file(path_candidate)
                if text:
                    return text
        return None

    def _file_paths(self, file: dict[str, Any]) -> list[str]:
        paths: list[str] = []
        for candidate in (
            file.get("path"),
            file.get("filepath"),
            file.get("local_path"),
            file.get("file", {}).get("path") if isinstance(file.get("file"), dict) else None,
            file.get("meta", {}).get("path") if isinstance(file.get("meta"), dict) else None,
        ):
            if isinstance(candidate, str) and candidate.strip():
                paths.append(candidate)
        return paths

    def _read_text_file(self, path_str: str) -> str | None:
        try:
            path = Path(path_str)
            if not path.exists() or not path.is_file():
                return None
            if path.suffix.lower() not in {".txt", ".md", ".text", ".log"}:
                return None
            try:
                return path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None

    def _extract_chunk_size(self, message: str) -> int | None:
        match = re.match(
            r"^(?:chunk\s+size|size)\s+(\d+)$",
            message.strip(),
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        return max(100, min(12000, int(match.group(1))))

    def _load_document(
        self,
        session: SessionState,
        text: str,
        source_name: str | None = None,
    ) -> None:
        cleaned = text.strip()
        session.source_text = cleaned
        session.chunks = self._chunk_text(cleaned, session.chunk_size)
        session.current_index = 0
        first_line = cleaned.splitlines()[0].strip() if cleaned else ""
        session.source_name = source_name or (first_line[:60] if first_line else "Loaded text")

    def _chunk_text(self, text: str, chunk_size: int) -> list[str]:
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not cleaned:
            return []
        chunks: list[str] = []
        start = 0
        min_break = max(120, chunk_size // 3)
        while start < len(cleaned):
            end = min(len(cleaned), start + chunk_size)
            if end < len(cleaned):
                preferred_end = self._find_natural_break(cleaned, start, end, min_break)
                if preferred_end is not None:
                    end = preferred_end
            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
        return chunks or [cleaned]

    def _find_natural_break(
        self,
        text: str,
        start: int,
        target_end: int,
        min_break: int,
    ) -> int | None:
        search_start = start + min_break
        if search_start >= target_end:
            return None

        window = text[start:target_end]
        paragraph_breaks = ["\n\n", "\n \n"]
        sentence_breaks = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
        clause_breaks = ["; ", ": ", ", "]
        word_breaks = [" "]

        for breakpoints, include_len in (
            (paragraph_breaks, 2),
            (sentence_breaks, 1),
            (clause_breaks, 1),
            (word_breaks, 1),
        ):
            candidate = self._last_break_in_window(
                window,
                breakpoints,
                minimum_offset=min_break,
                include_len=include_len,
            )
            if candidate is not None:
                return start + candidate

        return None

    def _last_break_in_window(
        self,
        window: str,
        breakpoints: list[str],
        minimum_offset: int,
        include_len: int,
    ) -> int | None:
        best = -1
        best_len = 0
        for marker in breakpoints:
            idx = window.rfind(marker)
            if idx >= minimum_offset and idx > best:
                best = idx
                best_len = include_len
        if best < 0:
            return None
        return best + best_len

    def _move(self, session: SessionState, step: int) -> None:
        if not session.chunks:
            return
        session.current_index = max(
            0, min(len(session.chunks) - 1, session.current_index + step)
        )

    def _jump(self, session: SessionState, index: int) -> None:
        if not session.chunks:
            return
        session.current_index = max(0, min(len(session.chunks) - 1, index))

    def _goto(self, session: SessionState, message: str) -> None:
        if not session.chunks:
            return
        match = re.search(r"(\d+)", message)
        if match:
            self._jump(session, int(match.group(1)) - 1)

    def _chunk_header(self, session: SessionState) -> str:
        total = len(session.chunks)
        current = session.current_index + 1 if total else 0
        return (
            f"Document: {session.source_name}\n"
            f"Current chunk number: {current}\n"
            f"Chunk: {current}/{total}\n"
            f"Chunk size: {session.chunk_size}"
        )

    def _current_chunk(self, session: SessionState) -> str:
        if not session.chunks:
            return ""
        chunk = session.chunks[session.current_index]
        if len(chunk) <= self.valves.max_preview_chars:
            return chunk
        return chunk[: self.valves.max_preview_chars].rstrip()

    def _chunk_only_message(self, session: SessionState) -> str:
        if not session.chunks:
            return "No document is loaded yet. Upload or paste a text document."
        return f"{self._chunk_header(session)}\n\n{self._current_chunk(session)}"

    def _user_request_with_chunk(self, user_message: str, session: SessionState) -> str:
        if not session.chunks:
            return user_message
        return (
            f"{user_message.strip()}\n\n"
            f"{self._chunk_header(session)}\n\n"
            f"{self._current_chunk(session)}"
        )

    def _status_prompt(self, session: SessionState) -> str:
        if not session.chunks:
            return f"No document is loaded yet. Current chunk size is {session.chunk_size}."
        return (
            f"{self._chunk_header(session)}\n\n"
            "Commands: show, next, prev, first, last, goto N, chunk size N, summarize, reset."
        )

    def _help_prompt(self) -> str:
        return (
            "Chunk Reader commands:\n"
            "- load followed by pasted text\n"
            "- load file\n"
            "- show\n"
            "- next\n"
            "- prev\n"
            "- first\n"
            "- last\n"
            "- goto N\n"
            "- restart N\n"
            "- start N\n"
            "- chunk size N\n"
            "- summarize\n"
            "- status\n"
            "- reset"
        )

    def _looks_like_document(self, message: str) -> bool:
        stripped = message.strip()
        if len(stripped) >= 400:
            return True
        if "\n" in stripped and len(stripped) >= 150:
            return True
        sentence_count = len(re.findall(r"[.!?](?:\s|$)", stripped))
        return len(stripped) >= 200 and sentence_count >= 3

    def _is_summarize_request(self, lower_message: str) -> bool:
        summary_phrases = (
            "summarize",
            "summarise",
            "summary",
            "tldr",
            "tl;dr",
            "what is this chunk about",
            "what is this about",
            "explain this chunk",
            "describe this chunk",
        )
        return any(phrase in lower_message for phrase in summary_phrases)

    async def _emit_status(self, emitter, description: str, done: bool = False) -> None:
        if emitter is None:
            return
        await emitter(
            {
                "type": "status",
                "data": {
                    "description": description,
                    "done": done,
                    "hidden": False,
                },
            }
        )
