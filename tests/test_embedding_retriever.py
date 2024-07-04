import logging
from typing import List

import pytest
from haystack import Document, default_to_dict
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from pymilvus import RRFRanker

from src.milvus_haystack import MilvusDocumentStore
from src.milvus_haystack.document_store import MilvusStoreError
from src.milvus_haystack.milvus_embedding_retriever import (
    MilvusEmbeddingRetriever,
    MilvusHybridRetriever,
    MilvusSparseEmbeddingRetriever,
)

logger = logging.getLogger(__name__)

DEFAULT_CONNECTION_ARGS = {
    "uri": "http://localhost:19530",
    # "uri": "./milvus_test.db",  # This uri works for Milvus Lite
}


class TestMilvusEmbeddingTests:
    @pytest.fixture
    def document_store(self) -> MilvusDocumentStore:
        return MilvusDocumentStore(
            connection_args=DEFAULT_CONNECTION_ARGS,
            consistency_level="Strong",
            drop_old=True,
        )

    def test_run(self, document_store: MilvusDocumentStore):
        documents = []
        doc = Document(
            content="A Foo Document",
            meta={
                "name": "name_0",
                "page": "100",
                "chapter": "intro",
                "number": 2,
                "date": "1969-07-21T20:17:40",
            },
            embedding=[-10.0] * 128,
        )
        documents.append(doc)
        document_store.write_documents(documents)
        retriever = MilvusEmbeddingRetriever(
            document_store,
        )
        query_embedding = [-10.0] * 128
        res = retriever.run(query_embedding)
        assert res["documents"] == documents

    def test_to_dict(self, document_store: MilvusDocumentStore):
        expected_dict = {
            "type": "src.milvus_haystack.document_store.MilvusDocumentStore",
            "init_parameters": {
                "collection_name": "HaystackCollection",
                "collection_description": "",
                "collection_properties": None,
                "connection_args": DEFAULT_CONNECTION_ARGS,
                "consistency_level": "Strong",
                "index_params": None,
                "search_params": None,
                "drop_old": True,
                "primary_field": "id",
                "text_field": "text",
                "vector_field": "vector",
                "sparse_vector_field": None,
                "sparse_index_params": None,
                "sparse_search_params": None,
                "partition_key_field": None,
                "partition_names": None,
                "replica_number": 1,
                "timeout": None,
            },
        }
        retriever = MilvusEmbeddingRetriever(document_store)
        result = retriever.to_dict()

        assert result["type"] == "src.milvus_haystack.milvus_embedding_retriever.MilvusEmbeddingRetriever"
        assert result["init_parameters"]["document_store"] == expected_dict
        assert result["init_parameters"]["filters"] is None
        assert result["init_parameters"]["top_k"] == 10

    def test_from_dict(self, document_store: MilvusDocumentStore):
        retriever_dict = {
            "type": "src.milvus_haystack.milvus_embedding_retriever.MilvusEmbeddingRetriever",
            "init_parameters": {
                "document_store": {
                    "type": "milvus_haystack.document_store.MilvusDocumentStore",
                    "init_parameters": {
                        "collection_name": "HaystackCollection",
                        "collection_description": "",
                        "collection_properties": None,
                        "connection_args": DEFAULT_CONNECTION_ARGS,
                        "consistency_level": "Strong",
                        "index_params": None,
                        "search_params": None,
                        "drop_old": True,
                        "primary_field": "id",
                        "text_field": "text",
                        "vector_field": "vector",
                        "sparse_vector_field": None,
                        "sparse_index_params": None,
                        "sparse_search_params": None,
                        "partition_key_field": None,
                        "partition_names": None,
                        "replica_number": 1,
                        "timeout": None,
                    },
                },
                "filters": None,
                "top_k": 10,
            },
        }

        retriever = MilvusEmbeddingRetriever(document_store)

        reconstructed_retriever = MilvusEmbeddingRetriever.from_dict(retriever_dict)
        for field in vars(reconstructed_retriever):
            if field.startswith("__"):
                continue
            elif field == "document_store":
                for doc_store_field in vars(document_store):
                    if doc_store_field.startswith("__") or doc_store_field == "alias":
                        continue
                    assert getattr(reconstructed_retriever.document_store, doc_store_field) == getattr(
                        document_store, doc_store_field
                    )
            else:
                assert getattr(reconstructed_retriever, field) == getattr(retriever, field)


class TestMilvusSparseEmbeddingTests:
    @pytest.fixture
    def document_store(self) -> MilvusDocumentStore:
        return MilvusDocumentStore(
            connection_args=DEFAULT_CONNECTION_ARGS,
            consistency_level="Strong",
            drop_old=True,
            sparse_vector_field="sparse_vector",
        )

    @pytest.fixture
    def documents(self) -> List[Document]:
        documents = []
        doc = Document(
            content="A Foo Document",
            meta={
                "name": "name_0",
                "page": "100",
                "chapter": "intro",
                "number": 2,
                "date": "1969-07-21T20:17:40",
            },
            embedding=[-10.0] * 128,
            sparse_embedding=SparseEmbedding(indices=[0, 1, 2], values=[1.0, 2.0, 3.0]),
        )
        documents.append(doc)
        return documents

    def test_run(self, document_store: MilvusDocumentStore, documents: List[Document]):
        document_store.write_documents(documents)
        retriever = MilvusSparseEmbeddingRetriever(
            document_store,
        )
        sparse_query_embedding = SparseEmbedding(indices=[0, 1, 2], values=[1.0, 2.0, 3.0])
        res = retriever.run(sparse_query_embedding)
        assert res["documents"] == documents

    def test_fail_without_sparse_field(self, documents: List[Document]):
        document_store = MilvusDocumentStore(
            connection_args=DEFAULT_CONNECTION_ARGS,
            consistency_level="Strong",
            drop_old=True,
            vector_field="vector",
            # Missing sparse_vector_field
        )
        document_store.write_documents(documents)
        retriever = MilvusSparseEmbeddingRetriever(
            document_store,
        )
        sparse_query_embedding = SparseEmbedding(indices=[0, 1, 2], values=[1.0, 2.0, 3.0])
        with pytest.raises(MilvusStoreError):
            retriever.run(
                query_sparse_embedding=sparse_query_embedding,
            )

    def test_to_dict(self, document_store: MilvusDocumentStore):
        expected_dict = {
            "type": "src.milvus_haystack.document_store.MilvusDocumentStore",
            "init_parameters": {
                "collection_name": "HaystackCollection",
                "collection_description": "",
                "collection_properties": None,
                "connection_args": DEFAULT_CONNECTION_ARGS,
                "consistency_level": "Strong",
                "index_params": None,
                "search_params": None,
                "drop_old": True,
                "primary_field": "id",
                "text_field": "text",
                "vector_field": "vector",
                "sparse_vector_field": "sparse_vector",
                "sparse_index_params": None,
                "sparse_search_params": None,
                "partition_key_field": None,
                "partition_names": None,
                "replica_number": 1,
                "timeout": None,
            },
        }
        retriever = MilvusSparseEmbeddingRetriever(document_store)
        result = retriever.to_dict()

        assert result["type"] == "src.milvus_haystack.milvus_embedding_retriever.MilvusSparseEmbeddingRetriever"
        assert result["init_parameters"]["document_store"] == expected_dict
        assert result["init_parameters"]["filters"] is None
        assert result["init_parameters"]["top_k"] == 10

    def test_from_dict(self, document_store: MilvusDocumentStore):
        retriever_dict = {
            "type": "src.milvus_haystack.milvus_embedding_retriever.MilvusSparseEmbeddingRetriever",
            "init_parameters": {
                "document_store": {
                    "type": "milvus_haystack.document_store.MilvusDocumentStore",
                    "init_parameters": {
                        "collection_name": "HaystackCollection",
                        "collection_description": "",
                        "collection_properties": None,
                        "connection_args": DEFAULT_CONNECTION_ARGS,
                        "consistency_level": "Strong",
                        "index_params": None,
                        "search_params": None,
                        "drop_old": True,
                        "primary_field": "id",
                        "text_field": "text",
                        "vector_field": "vector",
                        "sparse_vector_field": "sparse_vector",
                        "sparse_index_params": None,
                        "sparse_search_params": None,
                        "partition_key_field": None,
                        "partition_names": None,
                        "replica_number": 1,
                        "timeout": None,
                    },
                },
                "filters": None,
                "top_k": 10,
            },
        }

        retriever = MilvusSparseEmbeddingRetriever(document_store)

        reconstructed_retriever = MilvusSparseEmbeddingRetriever.from_dict(retriever_dict)
        for field in vars(reconstructed_retriever):
            if field.startswith("__"):
                continue
            elif field == "document_store":
                for doc_store_field in vars(document_store):
                    if doc_store_field.startswith("__") or doc_store_field == "alias":
                        continue
                    assert getattr(reconstructed_retriever.document_store, doc_store_field) == getattr(
                        document_store, doc_store_field
                    )
            else:
                assert getattr(reconstructed_retriever, field) == getattr(retriever, field)


class TestMilvusHybridTests:
    @pytest.fixture
    def document_store(self) -> MilvusDocumentStore:
        return MilvusDocumentStore(
            connection_args=DEFAULT_CONNECTION_ARGS,
            consistency_level="Strong",
            drop_old=True,
            vector_field="vector",
            sparse_vector_field="sparse_vector",
        )

    @pytest.fixture
    def documents(self) -> List[Document]:
        documents = []
        doc = Document(
            content="A Foo Document",
            meta={
                "name": "name_0",
                "page": "100",
                "chapter": "intro",
                "number": 2,
                "date": "1969-07-21T20:17:40",
            },
            embedding=[-10.0] * 128,
            sparse_embedding=SparseEmbedding(indices=[0, 1, 2], values=[1.0, 2.0, 3.0]),
        )
        documents.append(doc)
        return documents

    def test_run(self, document_store: MilvusDocumentStore, documents: List[Document]):
        document_store.write_documents(documents)
        retriever = MilvusHybridRetriever(
            document_store,
        )
        query_embedding = [-10.0] * 128
        sparse_query_embedding = SparseEmbedding(indices=[0, 1, 2], values=[1.0, 2.0, 3.0])
        res = retriever.run(
            query_embedding=query_embedding,
            query_sparse_embedding=sparse_query_embedding,
        )
        assert res["documents"] == documents

    def test_fail_without_sparse_field(self, documents: List[Document]):
        document_store = MilvusDocumentStore(
            connection_args=DEFAULT_CONNECTION_ARGS,
            consistency_level="Strong",
            drop_old=True,
            vector_field="vector",
            # Missing sparse_vector_field
        )
        document_store.write_documents(documents)
        retriever = MilvusHybridRetriever(
            document_store,
        )
        query_embedding = [-10.0] * 128
        sparse_query_embedding = SparseEmbedding(indices=[0, 1, 2], values=[1.0, 2.0, 3.0])
        with pytest.raises(MilvusStoreError):
            retriever.run(
                query_embedding=query_embedding,
                query_sparse_embedding=sparse_query_embedding,
            )

    def test_to_dict(self, document_store: MilvusDocumentStore):
        expected_dict = {
            "type": "src.milvus_haystack.document_store.MilvusDocumentStore",
            "init_parameters": {
                "collection_name": "HaystackCollection",
                "collection_description": "",
                "collection_properties": None,
                "connection_args": DEFAULT_CONNECTION_ARGS,
                "consistency_level": "Strong",
                "index_params": None,
                "search_params": None,
                "drop_old": True,
                "primary_field": "id",
                "text_field": "text",
                "vector_field": "vector",
                "sparse_vector_field": "sparse_vector",
                "sparse_index_params": None,
                "sparse_search_params": None,
                "partition_key_field": None,
                "partition_names": None,
                "replica_number": 1,
                "timeout": None,
            },
        }
        retriever = MilvusHybridRetriever(document_store)
        result = retriever.to_dict()

        assert result["type"] == "src.milvus_haystack.milvus_embedding_retriever.MilvusHybridRetriever"
        assert result["init_parameters"]["document_store"] == expected_dict
        assert result["init_parameters"]["filters"] is None
        assert result["init_parameters"]["top_k"] == 10
        assert result["init_parameters"]["reranker"] == default_to_dict(RRFRanker())

    def test_from_dict(self, document_store: MilvusDocumentStore):
        retriever_dict = {
            "type": "src.milvus_haystack.milvus_embedding_retriever.MilvusHybridRetriever",
            "init_parameters": {
                "document_store": {
                    "type": "milvus_haystack.document_store.MilvusDocumentStore",
                    "init_parameters": {
                        "collection_name": "HaystackCollection",
                        "collection_description": "",
                        "collection_properties": None,
                        "connection_args": DEFAULT_CONNECTION_ARGS,
                        "consistency_level": "Strong",
                        "index_params": None,
                        "search_params": None,
                        "drop_old": True,
                        "primary_field": "id",
                        "text_field": "text",
                        "vector_field": "vector",
                        "sparse_vector_field": "sparse_vector",
                        "sparse_index_params": None,
                        "sparse_search_params": None,
                        "partition_key_field": None,
                        "partition_names": None,
                        "replica_number": 1,
                        "timeout": None,
                    },
                },
                "filters": None,
                "top_k": 10,
                "reranker": {"type": "pymilvus.client.abstract.RRFRanker", "init_parameters": {}},
            },
        }

        retriever = MilvusHybridRetriever(document_store)

        reconstructed_retriever = MilvusHybridRetriever.from_dict(retriever_dict)
        for field in vars(reconstructed_retriever):
            if field.startswith("__"):
                continue
            elif field == "document_store":
                for doc_store_field in vars(document_store):
                    if doc_store_field.startswith("__") or doc_store_field == "alias":
                        continue
                    assert getattr(reconstructed_retriever.document_store, doc_store_field) == getattr(
                        document_store, doc_store_field
                    )
            elif field == "reranker":
                assert default_to_dict(getattr(reconstructed_retriever, field)) == default_to_dict(RRFRanker())
            else:
                assert getattr(reconstructed_retriever, field) == getattr(retriever, field)
