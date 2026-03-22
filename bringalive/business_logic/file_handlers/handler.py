from typing import Iterable, AnyStr
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    content: str

    def to_tuple(self):
        return (self.id, self.content)


class Handler:

    def __init__(self, path_to_file, chunk_size=1024, overlap_size=128, id_override=None):
        self.path_to_file = path_to_file
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.id_override = id_override
        self.content = None
        self.chunks: Iterable[Chunk] = []

    def prepare_id(self, chunk_start: int):
        return f"{self.path_to_file}_{chunk_start}_{self.chunk_size}"

    def load(self) -> "Handler":
        raise NotImplementedError("Implement Loading")

    def transform(self) -> "Handler":
        raise NotImplementedError("Implement transform")

    def get(self) -> Iterable[Chunk]:
        return self.chunks

    def size(self) -> int:
        return len(self.chunks)
