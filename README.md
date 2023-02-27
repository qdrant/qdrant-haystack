# qdrant-haystack

An integration of [Qdrant](https://qdrant.tech) vector database with [Haystack](https://haystack.deepset.ai/)
by [deepset](https://www.deepset.ai).

The library finally allows using Qdrant as a document store, and provides an in-place replacement
for any other vector embeddings store. Thus, you should expect any kind of application to be working
smoothly just by changing the provider to `QdrantDocumentStore`.

## Installation

`qdrant-haystack` might be installed as any other Python library, using pip or poetry:

```bash
pip install qdrant-haystack
```

```bash
poetry add qdrant-haystack
```

## Usage

Once installed, you can already start using `QdrantDocumentStore` as any other store that supports
embeddings.

```python
from qdrant_haystack import QdrantDocumentStore

document_store = QdrantDocumentStore(
    host="localhost",
    index="Document",
    embedding_dim=512,
    recreate_index=True,
)
```

The list of parameters accepted by `QdrantDocumentStore` is complementary to those used in the
official [Python Qdrant client](https://github.com/qdrant/qdrant_client).
