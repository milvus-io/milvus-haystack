# Milvus Document Store for Haystack

[![PyPI - Version](https://img.shields.io/pypi/v/milvus-haystack.svg)](https://pypi.org/project/milvus-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/milvus-haystack.svg)](https://pypi.org/project/milvus-haystack)

## Installation

```shell
pip install --upgrade pymilvus milvus-haystack
```

## Usage

By default, if you install the latest version of pymilvus, you don't need to start the milvus service manually.
Optionally, you
can [start the Milvus service by docker](https://milvus.io/docs/install_standalone-docker.md#Start-Milvus).

Use the `MilvusDocumentStore` in a Haystack pipeline as a quick start.

```python
from haystack import Document
from milvus_haystack import MilvusDocumentStore

document_store = MilvusDocumentStore(
    # If you have installed the latest version of pymilvus with milvus lite, you can use a local path as the uri without starting the milvus service.
    connection_args={"uri": "./milvus.db"},
    # Or, if you have started the milvus standalone service by docker, you can use the specified uri to connect to the service.
    # connection_args={"uri": "http://localhost:19530"},
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

## Dive deep usage

Prepare an OpenAI API key and set it as an environment variable:

```shell
export OPENAI_API_KEY=<your_api_key>
```

Here are the ways to

- Create the indexing Pipeline
- Create the retrieval pipeline
- Create the RAG pipeline

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
    # If you have installed the latest version of pymilvus with milvus lite, you can use a local path as the uri without starting the milvus service.
    connection_args={"uri": "./milvus.db"},
    # Or, if you have started the milvus standalone service by docker, you can use the specified uri to connect to the service.
    # connection_args={"uri": "http://localhost:19530"},
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

## License

`milvus-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.