import json
import subprocess
import os
import pandas as pd


MODELS = [
    "all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "google/embeddinggemma-300m",
    'Lajavaness/bilingual-embedding-small',
    'intfloat/multilingual-e5-small'
]

ROOT = "/Users/jing/Documents/RaShips/revelio_matching"
EMBEDDINGS_INPUTS_LEGEND_DIR = f"{ROOT}/embeddings_files/embeddings_inputs_legend.json"
PROCESSED_FOLDER = f'{ROOT}/Processed'
THRESHOLD = 1000

def get_all_position_files_records_number() -> int:

    cleaned_file_files = list()
    processed_files = os.listdir(PROCESSED_FOLDER)
    records_number = 0

    for file in processed_files:

        if "individual_position" in file:

            file = pd.read_parquet(f"{PROCESSED_FOLDER}/{file}")
            records_number += file.shape[0]
            del file

    #         if 'csv' in file:
    #             cleaned_file = pd.read_csv(f'{PROCESSED_FOLDER}/{file}')
    #         elif ".parquet" in file:
    #             cleaned_file = pd.read_parquet(f'{PROCESSED_FOLDER}/{file}')
    #         else:
    #             cleaned_file = pd.DataFrame()

    #         cleaned_file = pd.concat([cleaned_file, cleaned_file])
    #         del cleaned_file

    # cleaned_file.drop_duplicates(inplace=True)
    # cleaned_file.reset_index(drop = True, inplace=True)
    
    return records_number

with open(EMBEDDINGS_INPUTS_LEGEND_DIR) as file:
    embeddings_legend = json.load(file)

embeddings_legend_indexes = list(embeddings_legend.keys())
position_records_number = get_all_position_files_records_number()
pitchbook_records_number = pd.read_parquet(f'{PROCESSED_FOLDER}/pitchbook_company_cleaned.parquet').shape[0]
total_records_number = position_records_number + pitchbook_records_number

n_iterations = (total_records_number // THRESHOLD) + 1

for model in MODELS:

    for index in embeddings_legend_indexes:
            
        for i in range(n_iterations):

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