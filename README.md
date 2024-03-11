# Milvus Document Store for Haystack


## Installation

```console
pip install -e milvus-haystack
```

## Usage
First, to start up a Milvus service, follow the ['Start Milvus'](https://milvus.io/docs/install_standalone-docker.md#Start-Milvus) instructions in the documentation. 

Then, to use the `MilvusDocumentStore` in a Haystack pipeline"

```py
from haystack import Document
from milvus_haystack import MilvusDocumentStore

document_store = MilvusDocumentStore()
documents = [Document(
    content="A Foo Document",
    meta={"page": "100", "chapter": "intro"},
    embedding=[-10.0] * 128,
)]
document_store.write_documents(documents)
document_store.count_documents()  # 1
```

## License

`milvus-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.