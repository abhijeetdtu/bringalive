
from metaflow import FlowSpec, step, catch, Parameter
import chromadb
import pathlib
import os
from typing import Iterable
import sys
sys.path.append(
    str(pathlib.Path(__file__, "..", "..", "..").resolve().absolute()))


from bringalive.business_logic.file_handlers.text_handler import TextHandler  # noqa: E402
from bringalive.business_logic.file_handlers.handler import Chunk  # noqa: E402


class LoadDocumentsFlow(FlowSpec):
    """
    python -m bringalive.business_logic.load_documents_flow run \
        --path_to_documents /Users/abhijeetpokhriyal/code/bringalive/bringalive/documents
    """
    path_to_documents = Parameter("path_to_documents")
    db_name = Parameter("db_name", default="document_db")
    collection_name = Parameter("collection_name", default="books")

    supported_formats = Parameter(
        "supported_formats", default=".txt", separator=";")
    chunk_size = Parameter("chunk_size", default=1024)
    overlap_size = Parameter("overlap_size", default=128)

    only_these_files = Parameter(
        "only_these_files", default=None, separator=";")

    @step
    def start(self):
        print(self.supported_formats)

        self.document_collection = []
        for document in os.listdir(self.path_to_documents):
            if self.only_these_files:
                if document not in self.only_these_files:
                    continue
            for format in self.supported_formats:
                if document.find(format) > -1:
                    abs_path = str(pathlib.Path(
                        self.path_to_documents, document).resolve().absolute())
                    self.document_collection.append(abs_path)
                    break

        self.next(self.load_documents, foreach="document_collection")

    @step
    def load_documents(self):
        abs_path = self.input
        th = TextHandler(abs_path, chunk_size=self.chunk_size,
                         overlap_size=self.overlap_size)
        self.content = th.load().transform().get()
        self.next(self.process_document)

    @step
    def process_document(self, inputs):

        client = chromadb.PersistentClient(
            path=str(pathlib.Path(self.path_to_documents, self.db_name)))

        collection = client.get_or_create_collection(name=self.collection_name)
        all_documents: Iterable[Chunk] = []
        for document in inputs:
            all_documents.extend(document.content)
        ids, documents = zip(*[document.to_tuple()
                             for document in all_documents])
        ids = list(ids)
        documents = list(documents)
        collection.upsert(ids=ids, documents=documents)
        self.next(self.end)

    @step
    def end(self):
        print("processed documents")


if __name__ == "__main__":
    LoadDocumentsFlow()
