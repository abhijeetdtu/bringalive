from typing import Iterable
from tempfile import NamedTemporaryFile
import tempfile
import shutil
import os

from bringalive.business_logic.file_handlers.handler import Chunk
from bringalive.business_logic.file_handlers.text_handler import TextHandler
from bringalive.constants import PATHS, VECTORDB, TEXT_FILE_HANDLER

import chromadb


class Memory:

    def __init__(self, collection=VECTORDB.default_collection.value,
                 path_to_vector_db: str = PATHS.path_to_vectordb.value):
        self.client = chromadb.PersistentClient(
            path=path_to_vector_db)
        self.collection = self.client.get_or_create_collection(
            collection)

    def query(
            self, query,
            n_results=VECTORDB.default_n_results.value):
        return self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

    def query_documents(
            self, query,
            n_results=VECTORDB.default_n_results.value):
        results = self.query(query, n_results)
        return results["documents"][0]

    def add_memories(self, all_documents: Iterable[Chunk]):
        ids, documents = zip(*[document.to_tuple()
                             for document in all_documents])
        ids = list(ids)
        documents = list(documents)
        self.collection.upsert(ids=ids, documents=documents)

    def add_memories_from_text(
            self, text: str,
            chunk_size: int = TEXT_FILE_HANDLER.default_chunk_size.value,
            overlap_size: int = TEXT_FILE_HANDLER.default_overlap_size.value):
        file, path = tempfile.mkstemp()
        os.write(file, text.encode("utf-8"))
        os.close(file)
        all_documents = (
            TextHandler(
                path,
                chunk_size,
                overlap_size
            )
            .load()
            .transform()
            .get()
        )
        ids, documents = zip(*[document.to_tuple()
                               for document in all_documents])
        ids = list(ids)
        documents = list(documents)
        self.collection.upsert(ids=ids, documents=documents)
        os.remove(path)
