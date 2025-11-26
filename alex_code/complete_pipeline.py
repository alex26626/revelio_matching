import json
import subprocess


MODELS = [
    "all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "google/embeddinggemma-300m",
    'Lajavaness/bilingual-embedding-small',
    'intfloat/multilingual-e5-small'
]

ROOT = "/Users/jing/Documents/RaShips/revelio_matching"
EMBEDDINGS_INPUTS_LEGEND_DIR = f"{ROOT}/embeddings_files/embeddings_inputs_legend.json"

with open(EMBEDDINGS_INPUTS_LEGEND_DIR) as file:
    embeddings_legend = json.load(file)

embeddings_legend_indexes = list(embeddings_legend.keys())

for model in MODELS:

    for index in embeddings_legend_indexes:

        subprocess.run([
            "python3",
            "embedding_pipeline.py",
            "--model_name",
            f"{model}",
            "--columns_mappings_index",
            f"{index}"
        ])

        if "/" in model:
            safe_model_name = model.split("/")[-1]
        else:
            safe_model_name = model

        EMBEDDINGS_FILE_DIR = f"{ROOT}/embeddings_files/embeddings_{index}/embeddings_{index}_{safe_model_name}.json"

        subprocess.run([
            "python3",
            "matches_pipeline.py",
            "--embeddings_file_dir",
            f"{EMBEDDINGS_FILE_DIR}"
        ])