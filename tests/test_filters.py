import logging
from typing import List

import numpy as np
import pytest
from haystack import Document
from haystack.testing.document_store import FilterDocumentsTest, _random_embeddings

from src.milvus_haystack import MilvusDocumentStore

logger = logging.getLogger(__name__)


class TestMilvusFilters(FilterDocumentsTest):
    @pytest.fixture
    def filterable_docs(self) -> List[Document]:
        documents = []
        for i in range(3):
            documents.append(
                Document(
                    content=f"A Foo Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "page": "100",
                        "chapter": "intro",
                        "number": 2,
                        "date": "1969-07-21T20:17:40",
                    },
                    embedding=_random_embeddings(768),
                )
            )
            documents.append(
                Document(
                    content=f"A Bar Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "page": "123",
                        "chapter": "abstract",
                        "number": 100,
                        "date": "1972-12-11T19:54:58",
                    },
                    embedding=_random_embeddings(768),
                )
            )
            documents.append(
                Document(
                    content=f"A Foobar Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "page": "90",
                        "chapter": "conclusion",
                        "number": -10,
                        "date": "1989-11-09T17:53:00",
                    },
                    embedding=_random_embeddings(768),
                )
            )
        return documents

    @pytest.fixture
    def document_store(self) -> MilvusDocumentStore:
        return MilvusDocumentStore(
            connection_args={
                "host": "localhost",
                "port": "19530",
                "user": "",
                "password": "",
                "secure": False,
            },
            drop_old=True,
        )

    def assert_documents_are_equal(self, received: List[Document], expected: List[Document]):
        """
        Assert that two lists of Documents are equal.
        This is used in every test, if a Document Store implementation has a different behaviour
        it should override this method.

        This can happen for example when the Document Store sets a score to returned Documents.
        Since we can't know what the score will be, we can't compare the Documents reliably.
        """
        assert len(received) == len(expected)
        received.sort(key=lambda x: x.id)
        expected.sort(key=lambda x: x.id)
        for r, e in zip(received, expected):
            assert r.content == e.content
            assert r.meta == e.meta
            assert r.content_type == e.content_type
            assert r.blob == e.blob
            assert r.score == r.score
            assert np.allclose(np.array(r.embedding), np.array(e.embedding), atol=1e-4)

    @pytest.mark.skip(reason="Milvus doesn't support comparison with dataframe")
    def test_comparison_equal_with_dataframe(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Milvus doesn't support comparison with dataframe")
    def test_comparison_not_equal_with_dataframe(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Milvus doesn't support comparison with dataframe")
    def test_comparison_greater_than_with_dataframe(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Milvus doesn't support comparison with dataframe")
    def test_comparison_greater_than_equal_with_dataframe(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Milvus doesn't support comparison with dataframe")
    def test_comparison_less_than_with_dataframe(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Milvus doesn't support comparison with dataframe")
    def test_comparison_less_than_equal_with_dataframe(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Milvus doesn't support comparison with Dates")
    def test_comparison_greater_than_with_iso_date(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Milvus doesn't support comparison with Dates")
    def test_comparison_greater_than_equal_with_iso_date(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Milvus doesn't support comparison with Dates")
    def test_comparison_less_than_with_iso_date(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Milvus doesn't support comparison with Dates")
    def test_comparison_less_than_equal_with_iso_date(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Milvus doesn't support comparison with None")
    def test_comparison_equal_with_none(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Milvus doesn't support comparison with None")
    def test_comparison_not_equal_with_none(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Milvus doesn't support comparison with None")
    def test_comparison_greater_than_with_none(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Milvus doesn't support comparison with None")
    def test_comparison_greater_than_equal_with_none(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Milvus doesn't support comparison with None")
    def test_comparison_less_than_with_none(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Milvus doesn't support comparison with None")
    def test_comparison_less_than_equal_with_none(self, document_store, filterable_docs): ...
