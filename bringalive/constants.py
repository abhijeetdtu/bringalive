import enum


class PATHS(enum.Enum):

    path_to_vectordb = "/Users/abhijeetpokhriyal/code/bringalive/bringalive/documents/document_db"


class VECTORDB(enum.Enum):
    default_collection = "books"
    default_n_results = 4


class MODELS(enum.Enum):
    default_model = "deepseek-r1:8b"
    # default_model = "deepseek-r1:1.5b"
    # default_model = 'llama3.2:3b'


class TEXT_FILE_HANDLER(enum.Enum):
    default_chunk_size = 1028
    default_overlap_size = 128
