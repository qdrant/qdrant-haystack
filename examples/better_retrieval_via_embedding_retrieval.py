import logging

logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING
)
logging.getLogger("haystack").setLevel(logging.INFO)


from qdrant_haystack.document_stores import QdrantDocumentStore

document_store = QdrantDocumentStore(
    "localhost",
    prefer_grpc=True,
    timeout=120,
    # recreate_index=True,
)

from haystack.utils import (
    clean_wiki_text,
    convert_files_to_docs,
    fetch_archive_from_http,
)

doc_dir = "data/tutorial6"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt6.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# Convert files to dicts
docs = convert_files_to_docs(
    dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True
)

# Now, let's write the dicts containing documents to our DB.
document_store.write_documents(docs)

from haystack.nodes import EmbeddingRetriever

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_format="sentence_transformers",
)

document_store.update_embeddings(retriever)

from haystack.nodes import FARMReader

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

from haystack.pipelines import ExtractiveQAPipeline

pipe = ExtractiveQAPipeline(reader, retriever)

# You can configure how many candidates the reader and retriever shall return
# The higher top_k for retriever, the better (but also the slower) your answers.
prediction = pipe.run(
    query="Who created the Dothraki vocabulary?",
    params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}},
)

from haystack.utils import print_answers

print_answers(prediction, details="minimum")
