import yaml
from haystack import Pipeline

from qdrant_haystack import QdrantDocumentStore  # noqa


def test_qdrant_document_store_works_in_pipelines():
    pipeline_yaml_string = """
    version: ignore

    components:    # define all the building-blocks for Pipeline
      - name: DocumentStore
        type: QdrantDocumentStore
        params:
          location: ':memory:'
          index: 'Document'
          embedding_dim: 512
          recreate_index: True
      - name: Retriever
        type: BM25Retriever
        params:
          document_store: DocumentStore
          top_k: 5
      - name: Reader
        type: FARMReader
        params:
          model_name_or_path: deepset/roberta-base-squad2
          context_window_size: 500
          return_no_answer: true

    pipelines:
      - name: query    # a sample extractive-qa Pipeline
        nodes:
          - name: Retriever
            inputs: [Query]
          - name: Reader
            inputs: [Retriever]
    """

    pipeline = Pipeline.load_from_config(
        yaml.safe_load(pipeline_yaml_string), pipeline_name="query"
    )

    assert pipeline is not None
