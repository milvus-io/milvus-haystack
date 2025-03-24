from enum import Enum

VECTOR_FIELD = "vector"
SPARSE_VECTOR_FIELD = "sparse"
TEXT_FIELD = "text"
PRIMARY_FIELD = "id"


class EmbeddingMode(Enum):
    """
    Modes of vector embedding calculation.

    - ``BUILTIN_FUNCTION``: Use the built-in embedding functions provided by Milvus.
      See https://milvus.io/docs/manage-collections.md#Function for more details.
    - ``EMBEDDING_MODEL``: Use an embedding model in haystack.
      See https://docs.haystack.deepset.ai/docs/embedders for more details.
    """

    BUILTIN_FUNCTION = "builtin_function"
    EMBEDDING_MODEL = "embedding_model"
