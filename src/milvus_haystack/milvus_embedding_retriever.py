from typing import Any, Dict, List, Optional

from haystack import Document, component

from milvus_haystack import MilvusDocumentStore


@component
class MilvusEmbeddingRetriever:
    """
    A component for retrieving documents from an Milvus Document Store.
    """

    def __init__(self, document_store: MilvusDocumentStore, filters: Optional[Dict[str, Any]] = None, top_k: int = 10):
        """
        Initializes a new instance of the MilvusEmbeddingRetriever.

        :param document_store: A Milvus Document Store object used to retrieve documents.
        :param filters: A dictionary with filters to narrow down the search space (default is None).
        :param top_k: The maximum number of documents to retrieve (default is 10).
        """
        self.filters = filters
        self.top_k = top_k
        self.document_store = document_store

    @component.output_types(documents=List[Document])
    def run(self, query_embedding: List[float]) -> Dict[str, List[Document]]:
        """
        Retrieve documents from the `MilvusDocumentStore`, based on their dense embeddings.

        :param query_embedding: Embedding of the query.
        :return: List of Document similar to `query_embedding`.
        """
        docs = self.document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=self.filters,
            top_k=self.top_k,
        )
        return {"documents": docs}
