import pytest


def test_query_document(vectordb_collection):
    results = vectordb_collection.query(
        # Chroma will embed this for you
        query_texts=["where was gandhi born"],
        n_results=2  # how many results to return
    )
    print(results)
    assert len(results) > 2
