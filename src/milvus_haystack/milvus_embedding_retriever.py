# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Optional, List

from haystack import component, Document

from src.milvus_haystack import MilvusDocumentStore


@component
class MilvusEmbeddingRetriever:
    """
    A component for retrieving documents from an ExampleDocumentStore.
    """

    def __init__(self, document_store: MilvusDocumentStore, filters: Optional[Dict[str, Any]] = None, top_k: int = 10):
        """
        Create an ExampleRetriever component. Usually you pass some basic configuration
        parameters to the constructor.

        :param document_store: A Document Store object used to retrieve documents
        :param filters: A dictionary with filters to narrow down the search space (default is None).
        :param top_k: The maximum number of documents to retrieve (default is 10).

        :raises ValueError: If the specified top_k is not > 0.
        """
        self.filters = filters
        self.top_k = top_k
        self.document_store = document_store

    @component.output_types(documents=List[Document])
    def run(self, query_embedding: List[float]):
        """
        Retrieve documents from the `PineconeDocumentStore`, based on their dense embeddings.

        :param query_embedding: Embedding of the query.
        :returns: List of Document similar to `query_embedding`.
        """
        docs = self.document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=self.filters,
            top_k=self.top_k,
        )
        return {"documents": docs}

