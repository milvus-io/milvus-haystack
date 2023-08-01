# milvus-documentstore


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

ds = MilvusDocumentStore()
ds.write_documents([Document("Some Content")])
ds.get_all_documents()  # prints [<Document: {'content': 'foo', 'content_type': 'text', ...>]
```

## License

`milvus-documentstore` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.