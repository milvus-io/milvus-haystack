import pytest
from haystack import Document
from src.milvus_haystack import MilvusDocumentStore
from src.milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever


class TestMilvusEmbeddingTests:
    from milvus import MilvusServer
    milvus_server = MilvusServer()
    milvus_server.set_base_dir("test_milvus_base")
    milvus_server.listen_port = 19530
    try:
        milvus_server.stop()
    except:
        pass
    try:
        milvus_server.cleanup()
    except:
        pass
    try:
        milvus_server.start()
    except:
        pass

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
        retriever = MilvusEmbeddingRetriever(document_store, )
        query_embedding = [-10.0] * 128
        res = retriever.run(query_embedding)
        assert res["documents"] == doc
