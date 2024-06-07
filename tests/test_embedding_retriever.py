import logging

import pytest
from haystack import Document

from src.milvus_haystack import MilvusDocumentStore
from src.milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever

logger = logging.getLogger(__name__)


class TestMilvusEmbeddingTests:
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
                "connection_args": {"host": "localhost", "port": "19530", "user": "", "password": "", "secure": False},
                "consistency_level": "Session",
                "index_params": None,
                "search_params": None,
                "drop_old": True,
                "primary_field": "id",
                "text_field": "text",
                "vector_field": "vector",
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
                        "connection_args": {
                            "host": "localhost",
                            "port": "19530",
                            "user": "",
                            "password": "",
                            "secure": False,
                        },
                        "consistency_level": "Session",
                        "index_params": None,
                        "search_params": None,
                        "drop_old": True,
                        "primary_field": "id",
                        "text_field": "text",
                        "vector_field": "vector",
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
                    if doc_store_field.startswith("__"):
                        continue
                    assert getattr(reconstructed_retriever.document_store, doc_store_field) == getattr(
                        document_store, doc_store_field
                    )
            else:
                assert getattr(reconstructed_retriever, field) == getattr(retriever, field)
