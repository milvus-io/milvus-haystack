import copy
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from haystack import Document, default_from_dict, default_to_dict
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.document_stores.errors import DocumentStoreError
from haystack.document_stores.types import DuplicatePolicy
from haystack.errors import FilterError
from haystack.utils import Secret, deserialize_secrets_inplace
from pymilvus import AnnSearchRequest, MilvusException, RRFRanker
from pymilvus.client.abstract import BaseRanker

from milvus_haystack.filters import parse_filters

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
        drop_old: Optional[bool] = False,  # noqa: FBT002
        *,
        primary_field: str = "id",
        text_field: str = "text",
        vector_field: str = "vector",
        sparse_vector_field: Optional[str] = None,
        sparse_index_params: Optional[dict] = None,
        sparse_search_params: Optional[dict] = None,
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
        :param partition_key_field: Name of the partition key field. Defaults to None.
        :param partition_names: List of partition names. Defaults to None.
        :param replica_number: Number of replicas. Defaults to 1.
        :param timeout: Timeout in seconds. Defaults to None.
        """
        try:
            from pymilvus import Collection, utility
        except ImportError as err:
            err_msg = "Could not import pymilvus python package. Please install it with `pip install pymilvus`."
            raise ValueError(err_msg) from err

        # Default search params when one is not provided.
        self.default_search_params = {
            "GPU_IVF_FLAT": {"metric_type": "L2", "params": {"nprobe": 10}},
            "GPU_IVF_PQ": {"metric_type": "L2", "params": {"nprobe": 10}},
            "GPU_CAGRA": {"metric_type": "L2", "params": {"itopk_size": 128}},
            "IVF_FLAT": {"metric_type": "L2", "params": {"nprobe": 10}},
            "IVF_SQ8": {"metric_type": "L2", "params": {"nprobe": 10}},
            "IVF_PQ": {"metric_type": "L2", "params": {"nprobe": 10}},
            "HNSW": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_FLAT": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_SQ": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_PQ": {"metric_type": "L2", "params": {"ef": 10}},
            "IVF_HNSW": {"metric_type": "L2", "params": {"nprobe": 10, "ef": 10}},
            "ANNOY": {"metric_type": "L2", "params": {"search_k": 10}},
            "AUTOINDEX": {"metric_type": "L2", "params": {}},
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
        self._sparse_vector_field = sparse_vector_field
        self.sparse_index_params = sparse_index_params
        self.sparse_search_params = sparse_search_params
        self._partition_key_field = partition_key_field
        self.fields: List[str] = []
        self.partition_names = partition_names
        self.replica_number = replica_number
        self.timeout = timeout

        # Create the connection to the server
        if connection_args is None:
            self.connection_args = DEFAULT_MILVUS_CONNECTION
        self.alias = self._create_connection_alias(self.connection_args)  # type: ignore[arg-type]
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
        # Determine result metadata fields.
        output_fields = self.fields[:]

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

        from pymilvus import Collection, MilvusException

        documents_cp = deepcopy(documents)
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
            if doc.sparse_embedding is None:
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
        if empty_sparse_embedding and self._sparse_vector_field is not None:
            logger.warning(
                "You specified `sparse_vector_field`, but document has no sparse embedding. "
                "A dummy sparse embedding will be used, but this can AFFECT THE SEARCH RESULTS!!! "
                "Please calculate the sparse embedding in each document first, and then write them to Milvus Store."
            )

        embeddings = [doc.embedding for doc in documents_cp]
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

        # Dict to hold all insert columns
        insert_dict: Dict[str, List] = {
            self._text_field: texts,
            self._vector_field: embeddings,
            self._primary_field: ids,
        }
        if self._sparse_vector_field:
            insert_dict[self._sparse_vector_field] = sparse_embeddings

        # Collect the meta into the insert dict.
        if metas is not None:
            for d in metas:
                for key, value in d.items():
                    if key in self.fields:
                        insert_dict.setdefault(key, []).append(value)

        # Total insert count
        vectors: list = insert_dict[self._vector_field]
        total_count = len(vectors)

        batch_size = 1000
        wrote_ids = []
        if not isinstance(self.col, Collection):
            raise MilvusException(message="Collection is not initialized")
        for i in range(0, total_count, batch_size):
            # Grab end index
            end = min(i + batch_size, total_count)
            # Convert dict to list of lists batch for insertion
            insert_list = [insert_dict[x][i:end] for x in self.fields]
            # Insert into the collection.
            try:
                # res: Collection
                res = self.col.insert(insert_list, timeout=None, **kwargs)
                wrote_ids.extend(res.primary_keys)
            except MilvusException as err:
                logger.error("Failed to insert batch starting at entity: %s/%s", i, total_count)
                raise err
        self.col.flush()
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
        return default_from_dict(cls, data)

    def _create_connection_alias(self, connection_args: dict) -> str:
        """Create the connection to the Milvus server."""
        from pymilvus import MilvusException, connections

        connection_args_cp = copy.deepcopy(connection_args)
        # Grab the connection arguments that are used for checking existing connection
        host: str = connection_args_cp.get("host", None)
        port: Union[str, int] = connection_args_cp.get("port", None)
        address: str = connection_args_cp.get("address", None)
        uri: str = connection_args_cp.get("uri", None)
        user = connection_args_cp.get("user", None)
        token: Union[str, Secret] = connection_args_cp.get("token", None)
        password: Union[str, Secret] = connection_args_cp.get("password", None)

        # Order of use is host/port, uri, address
        if host is not None and port is not None:
            given_address = str(host) + ":" + str(port)
        elif uri is not None:
            if uri.startswith("https://"):
                given_address = uri.split("https://")[1]
            elif uri.startswith("http://"):
                given_address = uri.split("http://")[1]
            else:
                given_address = uri  # Milvus lite
        elif address is not None:
            given_address = address
        else:
            given_address = None
            logger.debug("Missing standard address type for reuse attempt")

        # User defaults to empty string when getting connection info
        if user is not None:
            tmp_user = user
        else:
            tmp_user = ""

        # If a valid address was given, then check if a connection exists
        if given_address is not None:
            for con in connections.list_connections():
                addr = connections.get_connection_addr(con[0])
                if (
                    con[1]
                    and ("address" in addr)
                    and (addr["address"] == given_address)
                    and ("user" in addr)
                    and (addr["user"] == tmp_user)
                ):
                    logger.debug("Using previous connection: %s", con[0])
                    return con[0]

        # Generate a new connection if one doesn't exist
        alias = uuid4().hex
        token = self._resolve_value(token)
        password = self._resolve_value(password)
        if token is not None:
            connection_args_cp["token"] = token
        if password is not None:
            connection_args_cp["password"] = password
        try:
            connections.connect(alias=alias, **connection_args_cp)
            logger.debug("Created new connection using: %s", alias)
            return alias
        except MilvusException as err:
            logger.error("Failed to create new connection using: %s", alias)
            raise err

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
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            MilvusException,
        )
        from pymilvus.orm.types import infer_dtype_bydata

        # Determine embedding dim
        dim = len(embeddings[0])
        fields = []
        # Determine meta schema
        if metas:
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
        fields.append(FieldSchema(self._text_field, DataType.VARCHAR, max_length=65_535))
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
        from pymilvus import Collection

        if isinstance(self.col, Collection):
            schema = self.col.schema
            for x in schema.fields:
                self.fields.append(x.name)

    def _create_index(self) -> None:
        """Create an index on the collection"""
        from pymilvus import Collection, MilvusException

        if isinstance(self.col, Collection) and self._get_index() is None:
            try:
                # If no index params, use a default HNSW based one
                if self.index_params is None:
                    self.index_params = {
                        "metric_type": "L2",
                        "index_type": "HNSW",
                        "params": {"M": 8, "efConstruction": 64},
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
                        self.sparse_index_params = {
                            "index_type": "SPARSE_INVERTED_INDEX",
                            "metric_type": "IP",
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
        from pymilvus import Collection

        if isinstance(self.col, Collection) and self.search_params is None:
            index = self._get_index()
            if index is not None:
                index_type: str = index["index_param"]["index_type"]
                metric_type: str = index["index_param"]["metric_type"]
                self.search_params = self.default_search_params[index_type]
                self.search_params["metric_type"] = metric_type

    def _get_index(self) -> Optional[Dict[str, Any]]:
        """Return the vector index information if it exists"""
        from pymilvus import Collection

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
        from pymilvus import Collection, utility
        from pymilvus.client.types import LoadState

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
        self, query_embedding: List[float], filters: Optional[Dict[str, Any]] = None, top_k: int = 10
    ) -> List[Document]:
        """Dense embedding retrieval"""
        if self.col is None:
            logger.debug("No existing collection to search.")
            return []

        # Determine result metadata fields.
        output_fields = self.fields[:]

        # Build expr.
        if not filters:
            expr = None
        else:
            expr = parse_filters(filters)

        # Perform the search.
        res = self.col.search(
            data=[query_embedding],
            anns_field=self._vector_field,
            param=self.search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields,
            timeout=None,
        )
        docs = self._parse_search_result(res, output_fields=output_fields)
        return docs

    def _sparse_embedding_retrieval(
        self, query_sparse_embedding: SparseEmbedding, filters: Optional[Dict[str, Any]] = None, top_k: int = 10
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
            self.sparse_search_params = {"metric_type": "IP"}

        # Determine result metadata fields.
        output_fields = self.fields[:]

        # Build expr.
        if not filters:
            expr = None
        else:
            expr = parse_filters(filters)

        # Perform the search.
        search_data = self._convert_sparse_to_dict(query_sparse_embedding)
        res = self.col.search(
            data=[search_data],
            anns_field=self._sparse_vector_field,
            param=self.sparse_index_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields,
            timeout=None,
        )
        docs = self._parse_search_result(res, output_fields=output_fields)
        return docs

    def _hybrid_retrieval(
        self,
        query_embedding: List[float],
        query_sparse_embedding: SparseEmbedding,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        reranker: Optional[BaseRanker] = None,
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
            self.sparse_search_params = {"metric_type": "IP"}

        # Determine result metadata fields.
        output_fields = self.fields[:]

        # Build expr.
        if not filters:
            expr = None
        else:
            expr = parse_filters(filters)

        dense_req = AnnSearchRequest([query_embedding], self._vector_field, self.search_params, limit=top_k, expr=expr)
        sparse_req = AnnSearchRequest(
            [self._convert_sparse_to_dict(query_sparse_embedding)],
            self._sparse_vector_field,
            self.sparse_search_params,
            limit=top_k,
            expr=expr,
        )

        # Search topK docs based on dense and sparse vectors and rerank.
        res = self.col.hybrid_search([dense_req, sparse_req], rerank=reranker, limit=top_k, output_fields=output_fields)
        docs = self._parse_search_result(res, output_fields=output_fields)
        return docs

    def _parse_search_result(self, result, output_fields=None) -> List[Document]:
        if output_fields is None:
            output_fields = self.fields[:]
        docs = []
        for res in result[0]:
            data = {x: res.entity.get(x) for x in output_fields}
            doc = self._parse_document(data)
            docs.append(doc)
        return docs

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
