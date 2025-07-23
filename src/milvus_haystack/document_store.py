import importlib
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

from haystack import Document, default_from_dict, default_to_dict
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.document_stores.errors import DocumentStoreError
from haystack.document_stores.types import DuplicatePolicy
from haystack.errors import FilterError
from haystack.utils import Secret, deserialize_secrets_inplace
from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    MilvusException,
    RRFRanker,
    utility,
)
from pymilvus.client.abstract import BaseRanker
from pymilvus.client.types import LoadState
from pymilvus.orm.types import infer_dtype_bydata

from milvus_haystack.filters import parse_filters
from milvus_haystack.function import BaseMilvusBuiltInFunction, BM25BuiltInFunction
from milvus_haystack.utils.constant import PRIMARY_FIELD, TEXT_FIELD, VECTOR_FIELD, EmbeddingMode

logger = logging.getLogger(__name__)


class MilvusStoreError(DocumentStoreError):
    pass


DEFAULT_MILVUS_CONNECTION = {
    "host": "localhost",
    "port": "19530",
    "user": "",
    "password": "",
    "secure": False,
}

MAX_LIMIT_SIZE = 10_000


class MilvusDocumentStore:
    """
    Milvus Document Store.
    """

    def __init__(
        self,
        collection_name: str = "HaystackCollection",
        collection_description: str = "",
        collection_properties: Optional[Dict[str, Any]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        consistency_level: str = "Session",
        index_params: Optional[dict] = None,
        search_params: Optional[dict] = None,
        *,
        drop_old: Optional[bool] = False,
        primary_field: str = PRIMARY_FIELD,
        text_field: str = TEXT_FIELD,
        vector_field: str = VECTOR_FIELD,
        enable_dynamic_field: bool = True,
        sparse_vector_field: Optional[str] = None,
        sparse_index_params: Optional[dict] = None,
        sparse_search_params: Optional[dict] = None,
        builtin_function: Optional[Union[BaseMilvusBuiltInFunction, List[BaseMilvusBuiltInFunction]]] = None,
        partition_key_field: Optional[str] = None,
        partition_names: Optional[list] = None,
        replica_number: int = 1,
        timeout: Optional[float] = None,
    ):
        """
        Initialize the Milvus vector store.
        For more information about Milvus, please refer to
        https://milvus.io/docs

        :param collection_name: The name of the collection to be created.
            "HaystackCollection" as default.
        :param collection_description: The description of the collection.
        :param collection_properties: The properties of the collection.
            Defaults to None.
            If set, will override collection existing properties.
            For example: {"collection.ttl.seconds": 60}.
        :param connection_args: The connection args used for this class comes in the form of a dict.
        - For the case of [Milvus Lite](https://milvus.io/docs/milvus_lite.md),
        the most convenient method, just set the uri as a local file.
            Examples:
                connection_args = {
                    "uri": "./milvus.db"
                }
        - For the case of Milvus server on [docker or kubernetes](https://milvus.io/docs/quickstart.md),
        it is recommended to use when you are dealing with large scale of data.
            Examples:
                connection_args = {
                    "uri": "http://localhost:19530"
                }
        - For the case of [Zilliz Cloud](https://zilliz.com/cloud), the fully managed
         cloud service for Milvus, adjust the uri and token, which correspond to the
         [Public Endpoint and Api key](https://docs.zilliz.com/docs/on-zilliz-cloud-console#free-cluster-details)
         in Zilliz Cloud.
             Examples:
                 connection_args = {
                     "uri": "https://in03-ba4234asae.api.gcp-us-west1.zillizcloud.com",  # Public Endpoint
                     "token": Secret.from_env_var("ZILLIZ_CLOUD_API_KEY"),  # API key.
                     "secure": True
                 }
        If you use `token` or `password`, we recommend using the `Secret` class to load
        the token from environment variable for security.
        :param consistency_level: The consistency level to use for a collection.
            Defaults to "Session".
        :param index_params: Which index params to use.
        :param search_params: Which search params to use. Defaults to default of index.
        :param drop_old: Whether to drop the current collection.
            Defaults to False.
        :param primary_field: Name of the primary key field. Defaults to "id".
        :param text_field: Name of the text field. Defaults to "text".
        :param vector_field: Name of the vector field. Defaults to "vector".
        :param enable_dynamic_field: Whether to enable dynamic field. Defaults to True.
            For more information about Milvus dynamic field,
            please refer to https://milvus.io/docs/enable-dynamic-field.md#Dynamic-Field
        :param sparse_vector_field: Name of the sparse vector field. Defaults to None,
            which means do not use sparse retrival,
            else enable sparse retrieval with this specified field.
            For more information about Milvus sparse retrieval,
            please refer to https://milvus.io/docs/sparse_vector.md#Sparse-Vector
        :param sparse_index_params: Which index params to use for sparse field.
            Only useful when `sparse_vector_field` is set.
            If not specified, will use a default value.
        :param sparse_search_params: Which search params to use for sparse field.
            Only useful when `sparse_vector_field` is set.
            If not specified, will use a default value.
        :param builtin_function: A list of built-in functions to use.
        :param partition_key_field: Name of the partition key field. Defaults to None.
        :param partition_names: List of partition names. Defaults to None.
        :param replica_number: Number of replicas. Defaults to 1.
        :param timeout: Timeout in seconds. Defaults to None.
        """
        # Default search params when one is not provided.
        self.default_search_params = {
            "FLAT": {"metric_type": "L2", "params": {}},
            "IVF_FLAT": {"metric_type": "L2", "params": {"nprobe": 10}},
            "IVF_SQ8": {"metric_type": "L2", "params": {"nprobe": 10}},
            "IVF_PQ": {"metric_type": "L2", "params": {"nprobe": 10}},
            "HNSW": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_FLAT": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_SQ": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_PQ": {"metric_type": "L2", "params": {"ef": 10}},
            "IVF_HNSW": {"metric_type": "L2", "params": {"nprobe": 10, "ef": 10}},
            "ANNOY": {"metric_type": "L2", "params": {"search_k": 10}},
            "SCANN": {"metric_type": "L2", "params": {"search_k": 10}},
            "AUTOINDEX": {"metric_type": "L2", "params": {}},
            "GPU_CAGRA": {"metric_type": "L2", "params": {"itopk_size": 128}},
            "GPU_IVF_FLAT": {"metric_type": "L2", "params": {"nprobe": 10}},
            "GPU_IVF_PQ": {"metric_type": "L2", "params": {"nprobe": 10}},
            "SPARSE_INVERTED_INDEX": {
                "metric_type": "IP",
                "params": {"drop_ratio_build": 0.2},
            },
            "SPARSE_WAND": {"metric_type": "IP", "params": {"drop_ratio_build": 0.2}},
        }

        self.collection_name = collection_name
        self.collection_description = collection_description
        self.collection_properties = collection_properties
        self.connection_args = connection_args
        self.consistency_level = consistency_level
        self.index_params = index_params
        self.search_params = search_params
        self.drop_old = drop_old

        self._primary_field = primary_field
        self._text_field = text_field
        self._vector_field = vector_field
        self.enable_dynamic_field = enable_dynamic_field
        self._sparse_vector_field = sparse_vector_field
        self.sparse_index_params = sparse_index_params
        self.sparse_search_params = sparse_search_params
        self._partition_key_field = partition_key_field
        self.fields: List[str] = []
        self.partition_names = partition_names
        self.replica_number = replica_number
        self.timeout = timeout
        self.builtin_function: List[BaseMilvusBuiltInFunction] = []
        if builtin_function:
            self.builtin_function = (
                [builtin_function] if isinstance(builtin_function, BaseMilvusBuiltInFunction) else builtin_function
            )
        self._check_function()

        # Create the connection to the server
        if connection_args is None:
            self.connection_args = DEFAULT_MILVUS_CONNECTION
        self._milvus_client = MilvusClient(
            **self.connection_args,
        )
        self.alias = self.client._using
        self.col: Optional[Collection] = None

        # Grab the existing collection if it exists
        if utility.has_collection(self.collection_name, using=self.alias):
            self.col = Collection(
                self.collection_name,
                using=self.alias,
            )
            if self.collection_properties is not None:
                self.col.set_properties(self.collection_properties)
        # If need to drop old, drop it
        if drop_old and isinstance(self.col, Collection):
            self.col.drop()
            self.col = None

        # Initialize the vector store
        self._init(
            partition_names=partition_names,
            replica_number=replica_number,
            timeout=timeout,
        )
        self._dummy_value = 999.0

    def _check_function(self):
        # In the future, we will support dense function after the Milvus's DIDO feature
        # is ready.
        # self._dense_mode = EmbeddingMode.EMBEDDING_MODEL
        self._sparse_mode = EmbeddingMode.EMBEDDING_MODEL
        # Check only one BM25BuiltInFunction in self.builtin_function
        if len([function for function in self.builtin_function if isinstance(function, BM25BuiltInFunction)]) > 1:
            error_msg = "Only one BM25BuiltInFunction is allowed"
            raise MilvusStoreError(error_msg)
        self._input_field_schema_kwargs = {}
        for function in self.builtin_function:
            # The `isinstance` check is not adapted to the unittest, so here we use the class name check.
            if function.__class__.__name__ == BM25BuiltInFunction.__name__:
                if self._text_field != function.input_field_names[0]:
                    error_msg = "BM25BuiltInFunction input_field_names must be the same as text_field"
                    raise MilvusStoreError(error_msg)
                if self._sparse_vector_field != function.output_field_names[0]:
                    error_msg = "BM25BuiltInFunction output_field_names must be the same as sparse_vector_field"
                    raise MilvusStoreError(error_msg)
                self._sparse_mode = EmbeddingMode.BUILTIN_FUNCTION
                self._input_field_schema_kwargs = function.get_input_field_schema_kwargs()

    @property
    def client(self) -> MilvusClient:
        """Get client."""
        return self._milvus_client

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.

        :return: The number of documents in the document store.
        """
        if self.col is None:
            logger.debug("No existing collection to count.")
            return 0
        count_expr = "count(*)"
        res = self.col.query(
            expr="",
            output_fields=[count_expr],
        )
        doc_num = res[0][count_expr]
        return doc_num

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        Filters are defined as nested dictionaries that can be of two types:
        - Comparison
        - Logic

        Comparison dictionaries must contain the keys:

        - `field`
        - `operator`
        - `value`

        Logic dictionaries must contain the keys:

        - `operator`
        - `conditions`

        The `conditions` key must be a list of dictionaries, either of type Comparison or Logic.

        The `operator` value in Comparison dictionaries must be one of:

        - `==`
        - `!=`
        - `>`
        - `>=`
        - `<`
        - `<=`
        - `in`
        - `not in`

        The `operator` values in Logic dictionaries must be one of:

        - `NOT`
        - `OR`
        - `AND`


        A simple filter:
        ```python
        filters = {"field": "meta.type", "operator": "==", "value": "article"}
        ```

        A more complex filter:
        ```python
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "article"},
                {"field": "meta.date", "operator": ">=", "value": 1420066800},
                {"field": "meta.date", "operator": "<", "value": 1609455600},
                {"field": "meta.rating", "operator": ">=", "value": 3},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.genre", "operator": "in", "value": ["economy", "politics"]},
                        {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
                    ],
                },
            ],
        }

        :param filters: The filters to apply to the document list.
        :return: A list of Documents that match the given filters.
        """
        if self.col is None:
            logger.debug("No existing collection to filter.")
            return []
        output_fields = self._get_output_fields()

        # Build expr.
        if not filters:
            expr = ""
        else:
            expr = parse_filters(filters)

        # Perform the Query.
        try:
            res = self.col.query(
                expr=expr,
                output_fields=output_fields,
                limit=MAX_LIMIT_SIZE,
            )
        except MilvusException as err:
            logger.error("Failed to query documents with filters expr: %s", expr)
            raise FilterError(err) from err
        docs = [self._parse_document(d) for d in res]
        return docs

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes documents into the store.

        :param documents: A list of documents.
        :param policy: Documents with the same ID count as duplicates.
            MilvusStore only supports `DuplicatePolicy.NONE`
        :return: Number of documents written.
        """

        documents_cp = [MilvusDocumentStore._discard_invalid_meta(doc) for doc in deepcopy(documents)]
        if len(documents_cp) > 0 and not isinstance(documents_cp[0], Document):
            err_msg = "param 'documents' must contain a list of objects of type Document"
            raise ValueError(err_msg)

        if policy not in [DuplicatePolicy.NONE]:
            logger.warning(
                f"MilvusStore only supports `DuplicatePolicy.NONE`, but got {policy}. "
                "Milvus does not currently check if entity primary keys are duplicates."
                "You are responsible for ensuring entity primary keys are unique, "
                "and if they aren't Milvus may contain multiple entities with duplicate primary keys."
            )

        # Check embeddings
        embedding_dim = 128
        for doc in documents_cp:
            if doc.embedding is not None:
                embedding_dim = len(doc.embedding)
                break
        empty_embedding = False
        empty_sparse_embedding = False
        for doc in documents_cp:
            if doc.embedding is None:
                empty_embedding = True
                dummy_vector = [self._dummy_value] * embedding_dim
                doc.embedding = dummy_vector
            if doc.sparse_embedding is None and self._sparse_mode == EmbeddingMode.EMBEDDING_MODEL:
                empty_sparse_embedding = True
                dummy_sparse_vector = SparseEmbedding(
                    indices=[0],
                    values=[self._dummy_value],
                )
                doc.sparse_embedding = dummy_sparse_vector
            if doc.content is None:
                doc.content = ""
        if empty_embedding and self._sparse_vector_field is None:
            logger.warning(
                "Milvus is a purely vector database, but document has no embedding. "
                "A dummy embedding will be used, but this can AFFECT THE SEARCH RESULTS!!! "
                "Please calculate the embedding in each document first, and then write them to Milvus Store."
            )
        if (
            empty_sparse_embedding
            and self._sparse_vector_field is not None
            and self._sparse_mode == EmbeddingMode.EMBEDDING_MODEL
        ):
            logger.warning(
                "You specified `sparse_vector_field`, but document has no sparse embedding. "
                "A dummy sparse embedding will be used, but this can AFFECT THE SEARCH RESULTS!!! "
                "Please calculate the sparse embedding in each document first, and then write them to Milvus Store."
            )

        embeddings = [doc.embedding for doc in documents_cp]
        sparse_embeddings = None
        if self._sparse_mode == EmbeddingMode.EMBEDDING_MODEL:
            sparse_embeddings = [self._convert_sparse_to_dict(doc.sparse_embedding) for doc in documents_cp]
        metas = [doc.meta for doc in documents_cp]
        texts = [doc.content for doc in documents_cp]
        ids = [doc.id for doc in documents_cp]

        if len(embeddings) == 0:
            logger.debug("Nothing to insert, skipping.")
            return 0

        # If the collection hasn't been initialized yet, perform all steps to do so
        kwargs: Dict[str, Any] = {}
        if not isinstance(self.col, Collection):
            kwargs = {"embeddings": embeddings, "metas": metas}
            if self.partition_names:
                kwargs["partition_names"] = self.partition_names
            if self.replica_number:
                kwargs["replica_number"] = self.replica_number
            if self.timeout:
                kwargs["timeout"] = self.timeout
            self._init(**kwargs)

        insert_list: list[dict] = []
        for i in range(len(ids)):
            entity_dict = {
                self._text_field: texts[i],
                self._vector_field: embeddings[i],
                self._primary_field: ids[i],
            }
            if (
                self._sparse_vector_field
                and self._sparse_mode == EmbeddingMode.EMBEDDING_MODEL
                and sparse_embeddings is not None
            ):
                entity_dict[self._sparse_vector_field] = sparse_embeddings[i]
            if metas is not None:
                for key, value in metas[i].items():
                    # if not enable_dynamic_field, skip fields not in the collection.
                    if not self.enable_dynamic_field and key not in self.fields:
                        continue
                    # If enable_dynamic_field, all fields are allowed.
                    entity_dict[key] = value
            insert_list.append(entity_dict)

        total_count = len(insert_list)
        batch_size = 1000
        wrote_ids = []
        if not isinstance(self.col, Collection):
            raise MilvusException(message="Collection is not initialized")
        for i in range(0, total_count, batch_size):
            # Grab end index
            end = min(i + batch_size, total_count)
            batch_insert_list = insert_list[i:end]
            # Insert into the collection.
            try:
                # res: Collection
                res = self.col.insert(batch_insert_list, timeout=None, **kwargs)
                wrote_ids.extend(res.primary_keys)
            except MilvusException as err:
                logger.error("Failed to insert batch starting at entity: %s/%s", i, total_count)
                raise err
        return len(wrote_ids)

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.

        :param document_ids: The object_ids to delete
        """
        if self.col is None:
            logger.debug("No existing collection to delete.")
            return None
        expr = "id in ['" + "','".join(document_ids) + "']"
        logger.info(expr)
        self.col.delete(expr)

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the document store.

        :return: A dictionary representation of the document store.
        """
        new_connection_args = {}
        for conn_arg_key, conn_arg_value in self.connection_args.items():  # type: ignore[union-attr]
            if isinstance(conn_arg_value, Secret):
                new_connection_args[conn_arg_key] = conn_arg_value.to_dict()
            else:
                new_connection_args[conn_arg_key] = conn_arg_value
        init_parameters = {
            "collection_name": self.collection_name,
            "collection_description": self.collection_description,
            "collection_properties": self.collection_properties,
            "connection_args": new_connection_args,
            "consistency_level": self.consistency_level,
            "index_params": self.index_params,
            "search_params": self.search_params,
            "drop_old": self.drop_old,
            "primary_field": self._primary_field,
            "text_field": self._text_field,
            "vector_field": self._vector_field,
            "sparse_vector_field": self._sparse_vector_field,
            "sparse_index_params": self.sparse_index_params,
            "sparse_search_params": self.sparse_search_params,
            "builtin_function": [func.to_dict() for func in self.builtin_function],
            "partition_key_field": self._partition_key_field,
            "partition_names": self.partition_names,
            "replica_number": self.replica_number,
            "timeout": self.timeout,
        }
        return default_to_dict(self, **init_parameters)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MilvusDocumentStore":
        """
        Creates a new document store from a dictionary.

        :param data: The dictionary to use to create the document store.
        :return: A new document store.
        """
        for conn_arg_key, conn_arg_value in data["init_parameters"]["connection_args"].items():
            if isinstance(conn_arg_value, dict) and "type" in conn_arg_value and conn_arg_value["type"] == "env_var":
                deserialize_secrets_inplace(data["init_parameters"]["connection_args"], keys=[conn_arg_key])

        if "builtin_function" in data["init_parameters"]:
            builtin_function = []
            for func_init_dict in data["init_parameters"]["builtin_function"]:
                func_type = func_init_dict["type"]
                func_params = func_init_dict["init_parameters"]

                # Import the function class dynamically
                module_name, class_name = func_type.rsplit(".", 1)
                module = importlib.import_module(module_name)
                func_class = getattr(module, class_name)

                # Instantiate the function with parameters
                func_instance = func_class(**func_params)
                builtin_function.append(func_instance)

            data["init_parameters"]["builtin_function"] = builtin_function

        return default_from_dict(cls, data)

    def _init(
        self,
        embeddings: Optional[List] = None,
        metas: Optional[List[Dict]] = None,
        partition_names: Optional[List] = None,
        replica_number: int = 1,
        timeout: Optional[float] = None,
    ) -> None:
        if embeddings is not None:
            self._create_collection(embeddings, metas)
        self._extract_fields()
        self._create_index()
        self._create_search_params()
        self._load(
            partition_names=partition_names,
            replica_number=replica_number,
            timeout=timeout,
        )

    def _create_collection(self, embeddings: list, metas: Optional[List[Dict]] = None) -> None:
        # Determine embedding dim
        dim = len(embeddings[0])
        fields = []
        if not self.enable_dynamic_field and metas:
            # Determine meta schema
            # Create FieldSchema for each entry in meta.
            for key, value in metas[0].items():
                # Infer the corresponding datatype of the meta
                dtype = infer_dtype_bydata(value)
                # Datatype isn't compatible
                if dtype in [DataType.UNKNOWN, DataType.NONE]:
                    err_msg = f"Failure to create collection, unrecognized dtype for key: {key}"
                    logger.error(err_msg)
                    raise ValueError(err_msg)
                # Datatype is a string/varchar equivalent
                elif dtype == DataType.VARCHAR:
                    fields.append(FieldSchema(key, DataType.VARCHAR, max_length=65_535))
                else:
                    fields.append(FieldSchema(key, dtype))

        # Create the text field
        fields.append(
            FieldSchema(self._text_field, DataType.VARCHAR, max_length=65_535, **self._input_field_schema_kwargs)
        )
        # Create the primary key field
        fields.append(FieldSchema(self._primary_field, DataType.VARCHAR, is_primary=True, max_length=65_535))
        # Create the vector field, supports binary or float vectors
        fields.append(FieldSchema(self._vector_field, infer_dtype_bydata(embeddings[0]), dim=dim))
        if self._sparse_vector_field:
            fields.append(FieldSchema(self._sparse_vector_field, DataType.SPARSE_FLOAT_VECTOR))

        # Create the schema for the collection
        schema = CollectionSchema(
            fields,
            description=self.collection_description,
            partition_key_field=self._partition_key_field,
            enable_dynamic_field=self.enable_dynamic_field,
            functions=[func.function for func in self.builtin_function],
        )

        # Create the collection
        try:
            self.col = Collection(
                name=self.collection_name,
                schema=schema,
                consistency_level=self.consistency_level,
                using=self.alias,
            )
            # Set the collection properties if they exist
            if self.collection_properties is not None:
                self.col.set_properties(self.collection_properties)
        except MilvusException as err:
            logger.error("Failed to create collection: %s error: %s", self.collection_name, err)
            raise err

    def _extract_fields(self) -> None:
        """Grab the existing fields from the Collection"""
        if isinstance(self.col, Collection):
            schema = self.col.schema
            for x in schema.fields:
                self.fields.append(x.name)

    def _create_index(self) -> None:
        """Create an index on the collection"""
        if isinstance(self.col, Collection) and self._get_index() is None:
            try:
                # If no index params, use a default AUTOINDEX based one
                if self.index_params is None:
                    self.index_params = {
                        "metric_type": "L2",
                        "index_type": "AUTOINDEX",
                        "params": {},
                    }

                try:
                    self.col.create_index(
                        self._vector_field,
                        index_params=self.index_params,
                        using=self.alias,
                    )

                # If default did not work, most likely on Zilliz Cloud
                except MilvusException:
                    # Use AUTOINDEX based index
                    self.index_params = {
                        "metric_type": "L2",
                        "index_type": "AUTOINDEX",
                        "params": {},
                    }
                    self.col.create_index(
                        self._vector_field,
                        index_params=self.index_params,
                        using=self.alias,
                    )
                if self._sparse_vector_field:
                    if self.sparse_index_params is None:
                        if self._sparse_mode == EmbeddingMode.EMBEDDING_MODEL:
                            self.sparse_index_params = {
                                "index_type": "SPARSE_INVERTED_INDEX",
                                "metric_type": "IP",
                            }
                        else:  # self._sparse_mode == EmbeddingMode.BUILTIN_FUNCTION:
                            self.sparse_index_params = {
                                "index_type": "AUTOINDEX",
                                "metric_type": "BM25",
                                "params": {},
                            }
                    self.col.create_index(
                        self._sparse_vector_field,
                        index_params=self.sparse_index_params,
                        using=self.alias,
                    )

                logger.debug(
                    "Successfully created an index on collection: %s",
                    self.collection_name,
                )

            except MilvusException as err:
                logger.error("Failed to create an index on collection: %s", self.collection_name)
                raise err

    def _create_search_params(self) -> None:
        """Generate search params based on the current index type"""
        if isinstance(self.col, Collection) and self.search_params is None:
            index = self._get_index()
            if index is not None:
                index_type: str = index["index_param"]["index_type"]
                metric_type: str = index["index_param"]["metric_type"]
                self.search_params = self.default_search_params[index_type]  # {"metric_type": "L2", "params": {}}
                self.search_params["metric_type"] = metric_type

    def _get_index(self) -> Optional[Dict[str, Any]]:
        """Return the vector index information if it exists"""
        if isinstance(self.col, Collection):
            for x in self.col.indexes:
                if x.field_name == self._vector_field:
                    return x.to_dict()
        return None

    def _load(
        self,
        partition_names: Optional[list] = None,
        replica_number: int = 1,
        timeout: Optional[float] = None,
    ) -> None:
        """Load the collection if available."""
        if (
            isinstance(self.col, Collection)
            and self._get_index() is not None
            and utility.load_state(self.collection_name, using=self.alias) == LoadState.NotLoad
        ):
            self.col.load(
                partition_names=partition_names,
                replica_number=replica_number,
                timeout=timeout,
            )

    def _resolve_value(self, secret: Union[str, Secret]):
        if isinstance(secret, Secret):
            return secret.resolve_value()
        if secret:
            logger.warning(
                "Some secret values are not encrypted. Please use `Secret` class to encrypt them. "
                "The best way to implement it is to use `Secret.from_env` to load from environment variables. "
                "For example:\n"
                "from haystack.utils import Secret\n"
                "token = Secret.from_env('YOUR_TOKEN_ENV_VAR_NAME')"
            )
        return secret

    def _embedding_retrieval(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        query_text: Optional[str] = None,
    ) -> List[Document]:
        """Dense embedding retrieval"""
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        output_fields = self._get_output_fields()

        # Build expr.
        if not filters:
            expr = None
        else:
            expr = parse_filters(filters)

        # Perform the search.
        search_data = self._prepare_search_data(
            query_text=query_text, query_embedding=query_embedding, field=self._vector_field
        )
        res = self.col.search(
            data=[search_data],
            anns_field=self._vector_field,
            param=self.search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields,
            timeout=None,
        )
        distance_to_score_fn = self._select_score_fn()
        docs = self._parse_search_result(res, distance_to_score_fn=distance_to_score_fn)
        return docs

    def _sparse_embedding_retrieval(
        self,
        query_sparse_embedding: SparseEmbedding,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        query_text: Optional[str] = None,
    ) -> List[Document]:
        """Sparse embedding retrieval"""
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []
        if self._sparse_vector_field is None:
            message = (
                "You need to specify `sparse_vector_field` in the document store "
                "to use sparse embedding retrieval. Such as: "
                "MilvusDocumentStore(..., sparse_vector_field='sparse_vector',...)"
            )
            raise MilvusStoreError(message)

        if self.sparse_search_params is None:
            if self._sparse_mode == EmbeddingMode.EMBEDDING_MODEL:
                self.sparse_search_params = {"metric_type": "IP"}
            else:  # self._sparse_mode == EmbeddingMode.BUILTIN_FUNCTION
                self.sparse_search_params = {"metric_type": "BM25"}

        output_fields = self._get_output_fields()

        # Build expr.
        if not filters:
            expr = None
        else:
            expr = parse_filters(filters)

        # Perform the search.
        search_data = self._prepare_search_data(
            query_text=query_text, query_embedding=query_sparse_embedding, field=self._sparse_vector_field
        )

        res = self.col.search(
            data=[search_data],
            anns_field=self._sparse_vector_field,
            param=self.sparse_search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields,
            timeout=None,
        )
        docs = self._parse_search_result(res)
        return docs

    def _hybrid_retrieval(
        self,
        query_embedding: List[float],
        query_sparse_embedding: Optional[SparseEmbedding] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        reranker: Optional[BaseRanker] = None,
        query_text: Optional[str] = None,
    ) -> List[Document]:
        """Hybrid retrieval using both dense and sparse embeddings"""
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []
        if self._sparse_vector_field is None:
            message = (
                "You need to specify `sparse_vector_field` in the document store "
                "to use hybrid retrieval. Such as: "
                "MilvusDocumentStore(..., sparse_vector_field='sparse_vector',...)"
            )
            raise MilvusStoreError(message)

        if reranker is None:
            reranker = RRFRanker()

        if self.sparse_search_params is None:
            if self._sparse_mode == EmbeddingMode.EMBEDDING_MODEL:
                self.sparse_search_params = {"metric_type": "IP"}
            else:  # self._sparse_mode == EmbeddingMode.BUILTIN_FUNCTION
                self.sparse_search_params = {"metric_type": "BM25"}

        output_fields = self._get_output_fields()

        # Build expr.
        if not filters:
            expr = None
        else:
            expr = parse_filters(filters)

        dense_search_data = self._prepare_search_data(
            query_text=query_text, query_embedding=query_embedding, field=self._vector_field
        )
        sparse_search_data = self._prepare_search_data(
            query_text=query_text, query_embedding=query_sparse_embedding, field=self._sparse_vector_field
        )
        dense_req = AnnSearchRequest(
            [dense_search_data], self._vector_field, self.search_params, limit=top_k, expr=expr
        )
        sparse_req = AnnSearchRequest(
            [sparse_search_data],
            self._sparse_vector_field,
            self.sparse_search_params,
            limit=top_k,
            expr=expr,
        )

        # Search topK docs based on dense and sparse vectors and rerank.
        res = self.col.hybrid_search([dense_req, sparse_req], rerank=reranker, limit=top_k, output_fields=output_fields)
        docs = self._parse_search_result(res)
        return docs

    def _parse_search_result(self, result, distance_to_score_fn=lambda x: x) -> List[Document]:
        docs = []
        for res in result[0]:
            data = {x: res.entity.get(x) for x in res.entity.fields}
            doc = self._parse_document(data)
            doc.score = distance_to_score_fn(res.distance)
            docs.append(doc)
        return docs

    def _select_score_fn(self):
        def _map_l2_to_similarity(l2_distance: float) -> float:
            """Return a similarity score on a scale [0, 1].
            It is recommended that the original vector is normalized,
            Milvus only calculates the value before applying square root.
            l2_distance range: (0 is most similar, 4 most dissimilar)
            See
            https://milvus.io/docs/metric.md?tab=floating#Euclidean-distance-L2
            """
            return 1 - l2_distance / 4.0

        def _map_ip_to_similarity(ip_score: float) -> float:
            """Return a similarity score on a scale [0, 1].
            It is recommended that the original vector is normalized,
            ip_score range: (1 is most similar, -1 most dissimilar)
            See
            https://milvus.io/docs/metric.md?tab=floating#Inner-product-IP
            https://milvus.io/docs/metric.md?tab=floating#Cosine-Similarity
            """
            return (ip_score + 1) / 2.0

        if not self.index_params:
            return lambda x: x
        metric_type = self.index_params.get("metric_type", None)
        if metric_type == "L2":
            return _map_l2_to_similarity
        elif metric_type in ["IP", "COSINE"]:
            return _map_ip_to_similarity
        else:
            return lambda x: x

    def _parse_document(self, data: dict) -> Document:
        # we store dummy vectors during writing documents if they are not provided,
        # so we don't return them if they are dummy vectors
        embedding = data.pop(self._vector_field)
        if all(x == self._dummy_value for x in embedding):
            embedding = None

        sparse_embedding = None
        sparse_dict = data.pop(self._sparse_vector_field, None)
        if sparse_dict:
            sparse_embedding = self._convert_dict_to_sparse(sparse_dict)
            if sparse_embedding.values == [self._dummy_value] and sparse_embedding.indices == [0]:
                sparse_embedding = None

        return Document(
            id=data.pop(self._primary_field),
            content=data.pop(self._text_field),
            embedding=embedding,
            sparse_embedding=sparse_embedding,
            meta=data,
        )

    def _convert_sparse_to_dict(self, sparse_embedding: SparseEmbedding) -> Dict:
        return dict(zip(sparse_embedding.indices, sparse_embedding.values))

    def _convert_dict_to_sparse(self, sparse_dict: Dict) -> SparseEmbedding:
        return SparseEmbedding(indices=list(sparse_dict.keys()), values=list(sparse_dict.values()))

    @staticmethod
    def _discard_invalid_meta(document: Document):
        """
        Remove metadata fields with unsupported types from the document.
        """
        if not isinstance(document, Document):
            msg = f"Invalid document type: {type(document)}"
            raise ValueError(msg)
        if document.meta:
            discarded_keys = []
            new_meta = {}
            for key, value in document.meta.items():
                dtype = infer_dtype_bydata(value)
                if dtype in (DataType.UNKNOWN, DataType.NONE):
                    discarded_keys.append(key)
                else:
                    new_meta[key] = value

            if discarded_keys:
                msg = (
                    f"Document {document.id} has metadata fields with unsupported types: {discarded_keys}. "
                    f"Supported types refer to Pymilvus DataType. The values of these fields will be discarded."
                )
                logger.warning(msg)
            document.meta = new_meta

        return document

    def _prepare_search_data(
        self,
        *,
        query_text: Optional[str],
        query_embedding: Optional[Union[List[float], SparseEmbedding]],
        field: Optional[str],
    ) -> Union[List[float], Dict, str]:
        search_data: Union[str, List[float], SparseEmbedding] = query_embedding
        if self._sparse_vector_field is not None and field == self._sparse_vector_field:
            if self._sparse_mode == EmbeddingMode.BUILTIN_FUNCTION:
                search_data = query_text
            else:  # self._sparse_mode == EmbeddingMode.EMBEDDING_MODEL
                if not isinstance(query_embedding, SparseEmbedding):
                    error_msg = "Query embedding must be a SparseEmbedding instance"
                    raise MilvusStoreError(error_msg)
                search_data = self._convert_sparse_to_dict(query_embedding)
        return search_data

    def _get_output_fields(self):
        if self.enable_dynamic_field:
            output_fields = ["*"]
            return output_fields

        output_fields = self.fields[:]
        if self._sparse_vector_field and self._sparse_mode == EmbeddingMode.BUILTIN_FUNCTION:
            output_fields.remove(self._sparse_vector_field)
        return output_fields
