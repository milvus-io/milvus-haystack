import logging
from dataclasses import fields
from typing import List

import numpy as np
import pytest
from haystack import Document, default_to_dict
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from pymilvus import RRFRanker

from src.milvus_haystack import MilvusDocumentStore
from src.milvus_haystack.document_store import MilvusStoreError
from src.milvus_haystack.function import BM25BuiltInFunction
from src.milvus_haystack.milvus_embedding_retriever import (
    MilvusEmbeddingRetriever,
    MilvusHybridRetriever,
    MilvusSparseEmbeddingRetriever,
)

logger = logging.getLogger(__name__)

DEFAULT_CONNECTION_ARGS = {
    "uri": "http://localhost:19530",  # This uri works for Milvus docker service
    # "uri": "./milvus_test.db",  # This uri works for Milvus Lite
}


def l2_normalization(x: List[float]) -> List[float]:
    v = np.array(x)
    l2_norm = np.linalg.norm(v)
    if l2_norm == 0:
        return np.zeros_like(v)
    normalized_v = v / l2_norm
    return normalized_v.tolist()


def assert_docs_equal_except_score(doc1: Document, doc2: Document):
    field_names = [field.name for field in fields(Document) if field.name != "score"]

    for field_name in field_names:
        value1 = getattr(doc1, field_name)
        value2 = getattr(doc2, field_name)
        assert value1 == value2


class TestMilvusEmbeddingTests:
    @pytest.fixture
    def document_store(self) -> MilvusDocumentStore:
        return MilvusDocumentStore(
            collection_name="TestMilvusEmbedding",
            connection_args=DEFAULT_CONNECTION_ARGS,
            consistency_level="Strong",
            drop_old=True,
        )

    def test_run(self, document_store: MilvusDocumentStore):
        documents = []
        for i in range(10):
            doc = Document(
                content="A Foo Document",
                meta={
                    "name": f"name_{i}",
                    "page": "100",
                    "chapter": "intro",
                    "number": 2,
                    "date": "1969-07-21T20:17:40",
                },
                embedding=l2_normalization([0.5] * 63 + [0.1 * i]),
            )
            documents.append(doc)
        document_store.write_documents(documents)
        retriever = MilvusEmbeddingRetriever(
            document_store,
        )
        query_embedding = l2_normalization([0.5] * 64)
        res = retriever.run(query_embedding)
        assert len(res["documents"]) == 10
        assert_docs_equal_except_score(res["documents"][0], documents[5])

    def test_to_dict(self, document_store: MilvusDocumentStore):
        expected_dict = {
            "type": "src.milvus_haystack.document_store.MilvusDocumentStore",
            "init_parameters": {
                "collection_name": "TestMilvusEmbedding",
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
                "builtin_function": [],
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
                        "collection_name": "TestMilvusEmbedding",
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
                        "builtin_function": [],
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
                    if doc_store_field.startswith("__") or doc_store_field in ["alias", "_milvus_client"]:
                        continue
                    if doc_store_field == "builtin_function":
                        for func, func_reconstructed in zip(
                            getattr(document_store, doc_store_field),
                            getattr(reconstructed_retriever.document_store, doc_store_field),
                        ):
                            for k, v in func.to_dict().items():
                                if k == "function_name":
                                    continue
                                assert v == func_reconstructed.to_dict()[k]
                    else:
                        assert getattr(reconstructed_retriever.document_store, doc_store_field) == getattr(
                            document_store, doc_store_field
                        )
            else:
                assert getattr(reconstructed_retriever, field) == getattr(retriever, field)


class TestMilvusSparseEmbeddingTests:
    @pytest.fixture
    def document_store(self) -> MilvusDocumentStore:
        return MilvusDocumentStore(
            collection_name="TestMilvusSparseEmbedding",
            connection_args=DEFAULT_CONNECTION_ARGS,
            consistency_level="Strong",
            drop_old=True,
            sparse_vector_field="sparse_vector",
        )

    @pytest.fixture
    def documents(self) -> List[Document]:
        documents = []
        for i in range(10):
            doc = Document(
                content="A Foo Document",
                meta={
                    "name": f"name_{i}",
                    "page": "100",
                    "chapter": "intro",
                    "number": 2,
                    "date": "1969-07-21T20:17:40",
                },
                embedding=l2_normalization([0.5] * 64),
                sparse_embedding=SparseEmbedding(indices=[0, 1, 2 + i], values=[1.0, 2.0, 3.0]),
            )
            documents.append(doc)
        return documents

    def test_run(self, document_store: MilvusDocumentStore, documents: List[Document]):
        document_store.write_documents(documents)
        retriever = MilvusSparseEmbeddingRetriever(
            document_store,
        )
        sparse_query_embedding = SparseEmbedding(indices=[0, 1, 2 + 5], values=[1.0, 2.0, 3.0])
        res = retriever.run(sparse_query_embedding)
        assert len(res["documents"]) == 10
        assert_docs_equal_except_score(res["documents"][0], documents[5])

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
                "collection_name": "TestMilvusSparseEmbedding",
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
                "builtin_function": [],
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
                        "collection_name": "TestMilvusSparseEmbedding",
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
                        "builtin_function": [],
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
                    if doc_store_field.startswith("__") or doc_store_field in ["alias", "_milvus_client"]:
                        continue
                    if doc_store_field == "builtin_function":
                        for func, func_reconstructed in zip(
                            getattr(document_store, doc_store_field),
                            getattr(reconstructed_retriever.document_store, doc_store_field),
                        ):
                            for k, v in func.to_dict().items():
                                if k == "function_name":
                                    continue
                                assert v == func_reconstructed.to_dict()[k]
                    else:
                        assert getattr(reconstructed_retriever.document_store, doc_store_field) == getattr(
                            document_store, doc_store_field
                        )
            else:
                assert getattr(reconstructed_retriever, field) == getattr(retriever, field)


class TestMilvusHybridTests:
    @pytest.fixture
    def document_store(self) -> MilvusDocumentStore:
        return MilvusDocumentStore(
            collection_name="TestMilvusHybrid",
            connection_args=DEFAULT_CONNECTION_ARGS,
            consistency_level="Strong",
            drop_old=True,
            vector_field="vector",
            sparse_vector_field="sparse_vector",
        )

    @pytest.fixture
    def documents(self) -> List[Document]:
        documents = []
        for i in range(10):
            doc = Document(
                content="A Foo Document",
                meta={
                    "name": f"name_{i}",
                    "page": "100",
                    "chapter": "intro",
                    "number": 2,
                    "date": "1969-07-21T20:17:40",
                },
                embedding=l2_normalization([0.5] * 63 + [0.45 + 0.01 * i]),
                sparse_embedding=SparseEmbedding(indices=[0, 1, 2 + i], values=[1.0, 2.0, 3.0]),
            )
            documents.append(doc)
        return documents

    def test_run(self, document_store: MilvusDocumentStore, documents: List[Document]):
        document_store.write_documents(documents)
        retriever = MilvusHybridRetriever(
            document_store,
        )
        query_embedding = l2_normalization([0.5] * 64)
        sparse_query_embedding = SparseEmbedding(indices=[0, 1, 2 + 5], values=[1.0, 2.0, 3.0])
        res = retriever.run(
            query_embedding=query_embedding,
            query_sparse_embedding=sparse_query_embedding,
        )
        assert len(res["documents"]) == 10
        assert_docs_equal_except_score(res["documents"][0], documents[5])

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
        query_embedding = l2_normalization([0.5] * 64)
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
                "collection_name": "TestMilvusHybrid",
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
                "builtin_function": [],
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
                        "collection_name": "TestMilvusHybrid",
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
                        "builtin_function": [],
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
                    if doc_store_field.startswith("__") or doc_store_field in ["alias", "_milvus_client"]:
                        continue
                    if doc_store_field == "builtin_function":
                        for func, func_reconstructed in zip(
                            getattr(document_store, doc_store_field),
                            getattr(reconstructed_retriever.document_store, doc_store_field),
                        ):
                            for k, v in func.to_dict().items():
                                if k == "function_name":
                                    continue
                                assert v == func_reconstructed.to_dict()[k]
                    else:
                        assert getattr(reconstructed_retriever.document_store, doc_store_field) == getattr(
                            document_store, doc_store_field
                        )
            elif field == "reranker":
                assert default_to_dict(getattr(reconstructed_retriever, field)) == default_to_dict(RRFRanker())
            else:
                assert getattr(reconstructed_retriever, field) == getattr(retriever, field)

    def test_run_with_topk(self, document_store: MilvusDocumentStore, documents: List[Document]):
        document_store.write_documents(documents)
        retriever = MilvusHybridRetriever(document_store)
        query_embedding = l2_normalization([0.5] * 64)
        sparse_query_embedding = SparseEmbedding(indices=[0, 1, 2 + 5], values=[1.0, 2.0, 3.0])
        res = retriever.run(
            query_embedding=query_embedding,
            query_sparse_embedding=sparse_query_embedding,
            top_k=5,
        )
        assert len(res["documents"]) == 5
        assert_docs_equal_except_score(res["documents"][0], documents[5])


class TestMilvusBuiltInFunction:
    @pytest.fixture
    def document_store(self) -> MilvusDocumentStore:
        document_store = MilvusDocumentStore(
            collection_name="TestMilvusBuiltInFunction",
            connection_args=DEFAULT_CONNECTION_ARGS,
            consistency_level="Strong",
            drop_old=True,
            text_field="text",
            vector_field="vector",
            sparse_vector_field="sparse",
            # sparse_search_params={
            #     "metric_type": "BM25",
            # },
            sparse_index_params={
                "index_type": "AUTOINDEX",  # "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
            },
            builtin_function=[
                BM25BuiltInFunction(
                    function_name="bm25_function",
                    input_field_names="text",
                    output_field_names="sparse",
                    # You can customize the analyzer_params and enable_match here.
                    # See https://milvus.io/docs/analyzer-overview.md for more details.
                    # analyzer_params=analyzer_params_custom,
                    # enable_match=True,
                )
            ],
        )
        return document_store

    @pytest.fixture
    def documents(self) -> List[Document]:
        documents = []
        for i in range(10):
            doc = Document(
                content=f"Foo Document{i}",
                meta={
                    "name": f"name_{i}",
                    "page": "100",
                    "chapter": "intro",
                    "number": 2,
                    "date": "1969-07-21T20:17:40",
                },
                embedding=l2_normalization([0.5] * 64),
            )
            documents.append(doc)
        return documents

    def test_run_sparse_with_builtin_bm25_function(
        self, document_store: MilvusDocumentStore, documents: List[Document]
    ):
        document_store.write_documents(documents)
        retriever = MilvusSparseEmbeddingRetriever(
            document_store,
        )
        res = retriever.run(query_text="Document5")
        assert_docs_equal_except_score(res["documents"][0], documents[5])

    def test_run_hybrid_with_builtin_bm25_function(
        self, document_store: MilvusDocumentStore, documents: List[Document]
    ):
        document_store.write_documents(documents)
        retriever = MilvusHybridRetriever(
            document_store,
        )
        query_embedding = l2_normalization([0.5] * 64)
        res = retriever.run(query_embedding, query_text="Document5")
        assert len(res["documents"]) == 10
        assert_docs_equal_except_score(res["documents"][0], documents[5])

    def test_to_dict(self, document_store: MilvusDocumentStore):
        expected_dict = {
            "type": "src.milvus_haystack.document_store.MilvusDocumentStore",
            "init_parameters": {
                "collection_name": "TestMilvusBuiltInFunction",
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
                "sparse_vector_field": "sparse",
                "sparse_index_params": {"index_type": "AUTOINDEX", "metric_type": "BM25"},
                "sparse_search_params": None,
                "builtin_function": [
                    {
                        "type": "src.milvus_haystack.function.BM25BuiltInFunction",
                        "init_parameters": {
                            "function_name": "bm25_function",
                            "input_field_names": "text",
                            "output_field_names": "sparse",
                            "analyzer_params": None,
                            "enable_match": False,
                        },
                    }
                ],
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
                        "collection_name": "TestMilvusBuiltInFunction",
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
                        "sparse_vector_field": "sparse",
                        "sparse_index_params": {"index_type": "AUTOINDEX", "metric_type": "BM25"},
                        "sparse_search_params": None,
                        "builtin_function": [
                            {
                                "type": "src.milvus_haystack.function.BM25BuiltInFunction",
                                "init_parameters": {
                                    "function_name": "bm25_function",
                                    "input_field_names": "text",
                                    "output_field_names": "sparse",
                                    "analyzer_params": None,
                                    "enable_match": False,
                                },
                            }
                        ],
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
                    if doc_store_field.startswith("__") or doc_store_field in ["alias", "_milvus_client"]:
                        continue
                    if doc_store_field == "builtin_function":
                        for func, func_reconstructed in zip(
                            getattr(document_store, doc_store_field),
                            getattr(reconstructed_retriever.document_store, doc_store_field),
                        ):
                            for k, v in func.to_dict().items():
                                if k == "function_name":
                                    continue
                                assert v == func_reconstructed.to_dict()[k]
                    else:
                        assert getattr(reconstructed_retriever.document_store, doc_store_field) == getattr(
                            document_store, doc_store_field
                        )
            elif field == "reranker":
                assert default_to_dict(getattr(reconstructed_retriever, field)) == default_to_dict(RRFRanker())
            else:
                assert getattr(reconstructed_retriever, field) == getattr(retriever, field)
