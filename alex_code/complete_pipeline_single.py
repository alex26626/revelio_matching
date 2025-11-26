import json
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name")
parser.add_argument("--columns_mappings_index")

cmdargs = parser.parse_args()
model = cmdargs.model_name
index = cmdargs.columns_mappings_index

ROOT = "/Users/jing/Documents/RaShips/revelio_matching"

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