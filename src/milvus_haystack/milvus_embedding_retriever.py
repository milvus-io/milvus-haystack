from typing import Any, Dict, List, Optional

from haystack import DeserializationError, Document, component, default_from_dict, default_to_dict

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the retriever component.

        :returns:
            A dictionary representation of the retriever component.
        """
        return default_to_dict(
            self, document_store=self.document_store.to_dict(), filters=self.filters, top_k=self.top_k
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MilvusEmbeddingRetriever":
        """
        Creates a new retriever from a dictionary.

        :param data: The dictionary to use to create the retriever.
        :return: A new retriever.
        """
        init_params = data.get("init_parameters", {})
        if "document_store" not in init_params:
            err_msg = "Missing 'document_store' in serialization data"
            raise DeserializationError(err_msg)

        docstore = MilvusDocumentStore.from_dict(init_params["document_store"])
        data["init_parameters"]["document_store"] = docstore

        return default_from_dict(cls, data)

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
