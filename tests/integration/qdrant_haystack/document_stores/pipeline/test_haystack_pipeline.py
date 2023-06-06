from pathlib import Path
from typing import Dict, Optional

from haystack import Pipeline
from haystack.nodes.base import BaseComponent
from haystack.pipelines.config import (
    get_component_definitions,
    get_pipeline_definition,
    read_pipeline_config_from_yaml,
)

BASE_DIR = Path(__file__).parent


class HaystackPipeline(Pipeline):
    @classmethod
    def load_from_config(
        cls,
        pipeline_config: Dict,
        pipeline_name: Optional[str] = None,
        overwrite_with_env_variables: bool = True,
        strict_version_check: bool = False,
    ):
        pipeline = cls()

        pipeline_definition = get_pipeline_definition(
            pipeline_config=pipeline_config, pipeline_name=pipeline_name
        )
        component_definitions = get_component_definitions(
            pipeline_config=pipeline_config,
            overwrite_with_env_variables=overwrite_with_env_variables,
        )
        components: Dict[str, BaseComponent] = {}
        for node_config in pipeline_definition["nodes"]:
            component = cls._load_or_get_component(
                name=node_config["name"],
                definitions=component_definitions,
                components=components,
            )
            pipeline.add_node(
                component=component,
                name=node_config["name"],
                inputs=node_config["inputs"],
            )

        pipeline.update_config_hash()
        return pipeline


def test_query_pipeline_has_access_to_qdrant():
    config = read_pipeline_config_from_yaml(BASE_DIR / "pipeline.yaml")

    pipelines = {}

    for p in config["pipelines"]:
        pipelines[p["name"]] = HaystackPipeline().load_from_yaml(
            path=BASE_DIR / "pipeline.yaml", pipeline_name=p["name"]
        )

    # pipelines["indexing_pipeline"].graph.nodes
    # pipelines["query_pipeline"].graph.nodes

    pipelines["indexing_pipeline"].run(file_paths=BASE_DIR / "documents.json")
    results = pipelines["query_pipeline"].run("what should i do if i lose my card?")
    print(results)
    assert False
