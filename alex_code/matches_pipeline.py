import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import json
import time
from typing import Tuple, List, Optional
import psutil
import argparse
import sys

from sentence_transformers import SentenceTransformer  # For sentence embeddings
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

print("Running file:", __file__)
print("cwd:", os.getcwd())

def get_all_position_files() -> pd.DataFrame:

    all_cleaned_positions = pd.DataFrame()
    processed_files = os.listdir(PROCESSED_FOLDER)

    for file in processed_files:

        if "individual_position" in file:

            if 'csv' in file:
                cleaned_file = pd.read_csv(f'{PROCESSED_FOLDER}/{file}')
            elif ".parquet" in file:
                cleaned_file = pd.read_parquet(f'{PROCESSED_FOLDER}/{file}')
            else:
                cleaned_file = pd.DataFrame()

            all_cleaned_positions = pd.concat([all_cleaned_positions, cleaned_file])
            del cleaned_file

    all_cleaned_positions.drop_duplicates(inplace=True)
    all_cleaned_positions.reset_index(drop = True, inplace=True)
    
    return all_cleaned_positions

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings_file_dir")

cmdargs = parser.parse_args()

EMBEDDINGS_FILE_DIR = cmdargs.embeddings_file_dir
ROOT = "/Users/jing/Documents/RaShips/revelio_matching"
EMBEDDINGS_LEGEND_DIR = f"{ROOT}/embeddings_files/embeddings_inputs_legend.json"
PROCESSED_FOLDER = f'{ROOT}/Processed'
OUTPUT_FOLDER = f"{'/'.join(EMBEDDINGS_FILE_DIR.split('/')[:-1])}/matches_output"

with open(EMBEDDINGS_FILE_DIR, 'r') as file:
    embeddings = json.load(file)

with open(EMBEDDINGS_LEGEND_DIR, 'r') as file:
    embeddings_legend = json.load(file)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

cleaned_pitchbook = pd.read_parquet(f"{PROCESSED_FOLDER}/pitchbook_company_cleaned.parquet")
all_cleaned_positions = get_all_position_files()
all_cleaned_positions['id'] = all_cleaned_positions.apply(lambda x: str(x['rcid']) + '_' + str(x.name), axis = 1)

pb_embeddings_first = embeddings.get('pitchbook')
pos_embeddings_first = embeddings.get('position')

UNWANTED_KEYS = {"length_text", "memory_GB"}

pb_filtered = {k: v for k, v in pb_embeddings_first.items() if k not in UNWANTED_KEYS}
pos_filtered = {k: v for k, v in pos_embeddings_first.items() if k not in UNWANTED_KEYS}

pb_embeddings = list(pb_filtered.values())
pos_embeddings = list(pos_filtered.values())

model_name = EMBEDDINGS_FILE_DIR.split('/')[-1].split('.')[0].split('_')[2]

columns_mapping_index = EMBEDDINGS_FILE_DIR.split('/')[-1].split('_')[1]
column_mapping = embeddings_legend[columns_mapping_index]
pb_column_mapping = column_mapping.get('pitchbook')
pos_column_mapping = column_mapping.get('position')

sim_matrix = cosine_similarity(pos_embeddings, pb_embeddings)
matches = dict()
matches['sim_score'] = []
i = 0
total = min(len(pos_embeddings), len(pb_embeddings))

with tqdm(total=total) as p_bar:

    while i < total:

        max_sim_score = sim_matrix[sim_matrix == sim_matrix.max()][0]
        max_coordinates = np.where(sim_matrix == np.max(sim_matrix))
        max_pos_coordinate = max_coordinates[0][0]
        max_pb_coordinate = max_coordinates[1][0]

        for column in pos_column_mapping.keys():
            if column not in all_cleaned_positions.columns:
                continue
            col_name = f'{column}_pos'
            if col_name not in matches:
                matches[col_name] = []
            matches[col_name].append(all_cleaned_positions.loc[max_pos_coordinate, column])


        for column in pb_column_mapping.keys():
            if column not in cleaned_pitchbook.columns:
                continue
            col_name = f'{column}_pb'
            if col_name not in matches:
                matches[col_name] = []
            matches[col_name].append(cleaned_pitchbook.loc[max_pb_coordinate, column])


        matches['sim_score'].append(max_sim_score)

        i += 1
        p_bar.update(1)

        sim_matrix[max_pos_coordinate, :] = 0
        sim_matrix[:, max_pb_coordinate] = 0

lengths = {k: len(v) for k,v in matches.items()}
print("Column lengths:", lengths)

matches_data = pd.DataFrame(matches)

matches_data.to_excel(f'{OUTPUT_FOLDER}/{model_name}_matches.xlsx', index=False)
