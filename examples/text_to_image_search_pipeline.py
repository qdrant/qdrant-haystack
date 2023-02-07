import os

from haystack import Document
from haystack.nodes.retriever.multimodal import MultiModalRetriever
from haystack.utils import fetch_archive_from_http
from haystack import Pipeline
from qdrant_haystack import QdrantDocumentStore


# Here we initialize the DocumentStore to store 512 dim image embeddings
# obtained using OpenAI CLIP model
document_store = QdrantDocumentStore(embedding_dim=512)


doc_dir = "data/tutorial19"

fetch_archive_from_http(
    url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/spirit-animals.zip",
    output_dir=doc_dir,
)


images = [
    Document(content=f"./{doc_dir}/spirit-animals/{filename}", content_type="image")
    for filename in os.listdir(f"./{doc_dir}/spirit-animals/")
]

document_store.write_documents(images)


retriever_text_to_image = MultiModalRetriever(
    document_store=document_store,
    query_embedding_model="sentence-transformers/clip-ViT-B-32",
    query_type="text",
    document_embedding_models={"image": "sentence-transformers/clip-ViT-B-32"},
)

# Now let's turn our images into embeddings and store them in the DocumentStore.
document_store.update_embeddings(retriever=retriever_text_to_image)


pipeline = Pipeline()
pipeline.add_node(
    component=retriever_text_to_image, name="retriever_text_to_image", inputs=["Query"]
)

results = pipeline.run(
    query="Animal that lives in the water",
    params={"retriever_text_to_image": {"top_k": 3}},
)

# Sort the results based on the scores
results = sorted(results["documents"], key=lambda d: d.score, reverse=True)

for doc in results:
    print(doc.score, doc.content)
