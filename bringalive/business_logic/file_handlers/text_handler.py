import re
from bringalive.business_logic.file_handlers.handler import Handler, Chunk


class TextHandler(Handler):

    def __init__(self, path_to_file, chunk_size, overlap_size, id_override=None):
        super().__init__(path_to_file, chunk_size, overlap_size, id_override)

    def load(self):
        with open(self.path_to_file, "r") as f:
            self.content = f.read()
        return self

    def transform(self):
        total_lines = len(self.content)
        i = 0
        while True:
            id = self.prepare_id(i)
            content = self.content[i: i+self.chunk_size]
            content = re.sub(r"[\r\n\t\s]{3+}", "", content)
            chunk = Chunk(id,  content)
            self.chunks.append(chunk)
            i += self.chunk_size - self.overlap_size
            if (i > total_lines) or (i < 0):
                break

        return self
