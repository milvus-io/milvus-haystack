# Milvus Document Store for Haystack

[![PyPI - Version](https://img.shields.io/pypi/v/milvus-haystack.svg)](https://pypi.org/project/milvus-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/milvus-haystack.svg)](https://pypi.org/project/milvus-haystack)

## Installation

```shell
pip install --upgrade pymilvus milvus-haystack
```

## Usage

Use the `MilvusDocumentStore` in a Haystack pipeline as a quick start.

```python
from haystack import Document
from milvus_haystack import MilvusDocumentStore

document_store = MilvusDocumentStore(
    connection_args={"uri": "./milvus.db"},
    drop_old=True,
)
documents = [Document(
    content="A Foo Document",
    meta={"page": "100", "chapter": "intro"},
    embedding=[-10.0] * 128,
)]
document_store.write_documents(documents)
print(document_store.count_documents())  # 1
```
### Different ways to connect to Milvus

- For the case of [Milvus Lite](https://milvus.io/docs/milvus_lite.md), the most convenient method, just set the uri as a local file.
```python
document_store = MilvusDocumentStore(
    connection_args={"uri": "./milvus.db"},
    drop_old=True,
)
```
                
- For the case of Milvus server on [docker or kubernetes](https://milvus.io/docs/quickstart.md), it is recommended to use when you are dealing with large scale of data. After starting the Milvus service, you can use the specified uri to connect to the service.
```python
document_store = MilvusDocumentStore(
    connection_args={"uri": "http://localhost:19530"},
    drop_old=True,
)
```

- For the case of [Zilliz Cloud](https://zilliz.com/cloud), the fully managed cloud service for Milvus, adjust the uri and token, which correspond to the [Public Endpoint and Api key](https://docs.zilliz.com/docs/on-zilliz-cloud-console#free-cluster-details) in Zilliz Cloud.
```python
from haystack.utils import Secret
document_store = MilvusDocumentStore(
    connection_args={
        "uri": "https://in03-ba4234asae.api.gcp-us-west1.zillizcloud.com",  # Your Public Endpoint
        "token": Secret.from_env_var("ZILLIZ_CLOUD_API_KEY"),  # API key, we recommend using the Secret class to load the token from env variable for security.
        "secure": True
    },
    drop_old=True,
)
```



## Dive deep usage

Prepare an OpenAI API key and set it as an environment variable:

```shell
export OPENAI_API_KEY=<your_api_key>
```

### Create the indexing Pipeline and index some documents

```python
import glob
import os

from haystack import Pipeline
from haystack.components.converters import MarkdownToDocument
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter

from milvus_haystack import MilvusDocumentStore
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever

current_file_path = os.path.abspath(__file__)
file_paths = [current_file_path]  # You can replace it with your own file paths.

document_store = MilvusDocumentStore(
    connection_args={"uri": "./milvus.db"},
    drop_old=True,
)
indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", MarkdownToDocument())
indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=2))
indexing_pipeline.add_component("embedder", OpenAIDocumentEmbedder())
indexing_pipeline.add_component("writer", DocumentWriter(document_store))
indexing_pipeline.connect("converter", "splitter")
indexing_pipeline.connect("splitter", "embedder")
indexing_pipeline.connect("embedder", "writer")
indexing_pipeline.run({"converter": {"sources": file_paths}})

print("Number of documents:", document_store.count_documents())
```

### Create the retrieval pipeline and try a query

```python
question = "How to set the service uri with milvus lite?"  # You can replace it with your own question. 

retrieval_pipeline = Pipeline()
retrieval_pipeline.add_component("embedder", OpenAITextEmbedder())
retrieval_pipeline.add_component("retriever", MilvusEmbeddingRetriever(document_store=document_store, top_k=3))
retrieval_pipeline.connect("embedder", "retriever")

retrieval_results = retrieval_pipeline.run({"embedder": {"text": question}})

for doc in retrieval_results["retriever"]["documents"]:
    print(doc.content)
    print("-" * 10)
```

### Create the RAG pipeline and try a query

```python
from haystack.utils import Secret
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator

prompt_template = """Answer the following query based on the provided context. If the context does
                     not include an answer, reply with 'I don't know'.\n
                     Query: {{query}}
                     Documents:
                     {% for doc in documents %}
                        {{ doc.content }}
                     {% endfor %}
                     Answer: 
                  """

rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", OpenAITextEmbedder())
rag_pipeline.add_component("retriever", MilvusEmbeddingRetriever(document_store=document_store, top_k=3))
rag_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
rag_pipeline.add_component("generator", OpenAIGenerator(api_key=Secret.from_token(os.getenv("OPENAI_API_KEY")),
                                                        generation_kwargs={"temperature": 0}))
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "generator")

results = rag_pipeline.run(
    {
        "text_embedder": {"text": question},
        "prompt_builder": {"query": question},
    }
)
print('RAG answer:', results["generator"]["replies"][0])

```

## Sparse Retrieval
```python
from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseDocumentEmbedder,
    FastembedSparseTextEmbedder,
)

from milvus_haystack import MilvusDocumentStore, MilvusSparseEmbeddingRetriever

document_store = MilvusDocumentStore(
    connection_args={"uri": "./milvus.db"},
    sparse_vector_field="sparse_vector",  # Specify a name of the sparse vector field to enable sparse retrieval.
    drop_old=True,
)

documents = [
    Document(content="My name is Wolfgang and I live in Berlin"),
    Document(content="I saw a black horse running"),
    Document(content="Germany has many big cities"),
    Document(content="fastembed is supported by and maintained by Milvus."),
]

sparse_document_embedder = FastembedSparseDocumentEmbedder()
writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("sparse_document_embedder", sparse_document_embedder)
indexing_pipeline.add_component("writer", writer)
indexing_pipeline.connect("sparse_document_embedder", "writer")

indexing_pipeline.run({"sparse_document_embedder": {"documents": documents}})

query_pipeline = Pipeline()
query_pipeline.add_component("sparse_text_embedder", FastembedSparseTextEmbedder())
query_pipeline.add_component("sparse_retriever", MilvusSparseEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("sparse_text_embedder.sparse_embedding", "sparse_retriever.query_sparse_embedding")

query = "Who supports fastembed?"

result = query_pipeline.run({"sparse_text_embedder": {"text": query}})

print(result["sparse_retriever"]["documents"][0])  # noqa: T201

# Document(id=..., content: 'fastembed is supported by and maintained by Milvus.', sparse_embedding: vector with 48 non-zero elements)
```

## Hybrid Retrieval

```python
from haystack import Document, Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseDocumentEmbedder,
    FastembedSparseTextEmbedder,
)

from milvus_haystack import MilvusDocumentStore, MilvusHybridRetriever

document_store = MilvusDocumentStore(
    connection_args={"uri": "./milvus.db"},
    drop_old=True,
    sparse_vector_field="sparse_vector",  # Specify a name of the sparse vector field to enable hybrid retrieval.
)

documents = [
    Document(content="My name is Wolfgang and I live in Berlin"),
    Document(content="I saw a black horse running"),
    Document(content="Germany has many big cities"),
    Document(content="fastembed is supported by and maintained by Milvus."),
]

writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("sparse_doc_embedder", FastembedSparseDocumentEmbedder())
indexing_pipeline.add_component("dense_doc_embedder", OpenAIDocumentEmbedder())
indexing_pipeline.add_component("writer", writer)
indexing_pipeline.connect("sparse_doc_embedder", "dense_doc_embedder")
indexing_pipeline.connect("dense_doc_embedder", "writer")

indexing_pipeline.run({"sparse_doc_embedder": {"documents": documents}})

querying_pipeline = Pipeline()
querying_pipeline.add_component("sparse_text_embedder",
                                FastembedSparseTextEmbedder(model="prithvida/Splade_PP_en_v1"))

querying_pipeline.add_component("dense_text_embedder", OpenAITextEmbedder())
querying_pipeline.add_component(
    "retriever",
    MilvusHybridRetriever(
        document_store=document_store,
        # reranker=WeightedRanker(0.5, 0.5),  # Default is RRFRanker()
    )
)

querying_pipeline.connect("sparse_text_embedder.sparse_embedding", "retriever.query_sparse_embedding")
querying_pipeline.connect("dense_text_embedder.embedding", "retriever.query_embedding")

question = "Who supports fastembed?"

results = querying_pipeline.run(
    {"dense_text_embedder": {"text": question},
     "sparse_text_embedder": {"text": question}}
)

print(results["retriever"]["documents"][0])

# Document(id=..., content: 'fastembed is supported by and maintained by Milvus.', embedding: vector of size 1536, sparse_embedding: vector with 48 non-zero elements)

```
## License

`milvus-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.