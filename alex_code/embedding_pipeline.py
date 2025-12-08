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

parser = argparse.ArgumentParser()
parser.add_argument("--model_name")
parser.add_argument("--columns_mappings_index")

cmdargs = parser.parse_args()
MODEL_NAME = cmdargs.model_name
COLUMN_MAPPINGS_INDEX = cmdargs.columns_mappings_index

# Folders directories
ROOT = "/Users/jing/Documents/RaShips/revelio_matching"
EMBEDDINGS_FILES_FOLDER = f"{ROOT}/embeddings_files/embeddings_{COLUMN_MAPPINGS_INDEX}"
PROCESSED_FOLDER = f'{ROOT}/Processed'
OUTPUT_FOLDER = f'{ROOT}/alex_outputs'
EMBEDDINGS_FILES_LEGEND = f"{ROOT}/embeddings_files/embeddings_inputs_legend.json"

THRESHOLD = 1000

def print_memory():
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1e9
    tqdm.write(f"Current memory usage: {mem_gb:.2f} GB")
    return mem_gb

def get_all_position_files() -> list[str]:

    cleaned_file_files = list()
    processed_files = os.listdir(PROCESSED_FOLDER)

    for file in processed_files:

        if "individual_position" in file:

            cleaned_file_files.append(file)

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
    
    return cleaned_file_files

def generate_embedding_inputs(columns_mappings: dict, data: pd.DataFrame, shift: int = 0) -> list[str]:

    embedding_inputs = []

    for i in tqdm(range(data.shape[0])):
        embedding_input = ''
        for column, embedding_text in columns_mappings.items():
            feature_to_insert = data.loc[i + shift, column]
            if pd.notna(feature_to_insert):
                embedding_input += f'{embedding_text}: {feature_to_insert}, '
        embedding_inputs.append(embedding_input)

    return embedding_inputs

def append_to_json(file_path: str, new_data: dict, dataset: str):
    """Safely append new embeddings to a JSON file by entity type."""

    # --- Step 1: Load existing JSON safely ---
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                all_data = json.load(f)
        except json.JSONDecodeError:
            tqdm.write(f"⚠️ Corrupted JSON file detected: {file_path}. Starting fresh.")
            all_data = {}
    else:
        all_data = {}

    # --- Step 2: Ensure the structure exists ---
    if dataset not in all_data:
        all_data[dataset] = {}

    existing_data = all_data[dataset]

    tqdm.write(f"📦 Existing {dataset} entries: {len(existing_data)}")

    # --- Step 3: Merge new data ---
    added = 0
    for key, value in new_data.get(dataset, {}).items():
        if key not in existing_data:
            existing_data[key] = value
            added += 1

    all_data[dataset] = existing_data

    tqdm.write(f"➕ Added {added} new {dataset} entries")

    # --- Step 4: Atomic write to prevent corruption ---
    tmp_path = f"{file_path}.tmp"
    with open(tmp_path, "w") as tmp:
        json.dump(all_data, tmp, indent=2)
    os.replace(tmp_path, file_path)

    tqdm.write(f"✅ JSON updated and saved to {file_path}")

def chunk_text(text: str, tokenizer, max_tokens: float) -> Tuple[List[str], float]:

    len_text = len(text)
    words = text.split()
    chunks, current_chunk = [], []
    current_length = 0

    for word in words:
        word_len = len(tokenizer.tokenize(word))
        if current_length + word_len > max_tokens:
            chunk_string = " ".join(current_chunk)
            if MODEL_NAME == "intfloat/multilingual-e5-small":
                chunk_string = f"query: {chunk_string}"
            chunks.append(chunk_string)
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += word_len
    if current_chunk:
        chunk_string = " ".join(current_chunk)
        if MODEL_NAME == "intfloat/multilingual-e5-small":
            chunk_string = f"query: {chunk_string}"
        chunks.append(chunk_string)
    return chunks, len_text

# Build general embeddings function

def generate_embeddings(
    embeddings_dict: dict, 
    dataset: str, 
    dir_list: List[str], 
    texts_list: List[str], 
    model: SentenceTransformer, 
    model_name: str,
    file_name: str,
) -> dict:
    
    tqdm.write(f"🚀 Working on {dataset} with model {model_name}")

    # Handle special models requiring trust_remote_code
    trust_remote_code = model_name in [
        "Lajavaness/bilingual-embedding-small",
        "Alibaba-NLP/gte-multilingual-base"
    ]

    # Clean model name for filename
    safe_model_name = model_name.split('/')[-1]
    output_name = f"{ROOT}/embeddings_files/embeddings_{COLUMN_MAPPINGS_INDEX}/{file_name}_{safe_model_name}.json"
    tokenizer = model.tokenizer
    max_tokens = tokenizer.model_max_length
    embeddings_counter = 0

    # --- Load existing embeddings to resume ---
    existing_dirs = set()
    if os.path.exists(output_name):
        try:
            with open(output_name, "r") as file:
                current_embeddings = json.load(file)
            existing_dirs = set(current_embeddings.get(dataset, {}).keys())
            tqdm.write(f"📂 Found existing {len(existing_dirs)} {dataset} embeddings. Skipping duplicates.")
        except json.JSONDecodeError:
            tqdm.write(f"⚠️ Corrupted JSON detected at {output_name}, starting fresh.")
            current_embeddings = {}
    else:
        current_embeddings = {}

    # Ensure structure
    if dataset not in embeddings_dict:
        embeddings_dict[dataset] = {}

    # --- Main loop ---
    for i in tqdm(range(len(dir_list)), desc=f"Embedding {dataset}"):
        text_complete = texts_list[i]
        dir_id = dir_list[i]

        # Skip if already embedded
        if dir_id in existing_dirs:
            continue

        if not (pd.notna(text_complete) and isinstance(text_complete, str) and text_complete.strip()):
            continue

        # --- Process text ---
        start = time.time()
        text_chunks, len_text = chunk_text(text=text_complete, tokenizer=tokenizer, max_tokens=max_tokens)
        #embeddings_dict[dataset]['length_text'] = len_text

        tqdm.write(f"📝 Text length: {len_text} chars")

        if trust_remote_code:
            entity_embedding = model.encode(
                text_chunks,
                convert_to_tensor=False,
                trust_remote_code=trust_remote_code)
        else:
            entity_embedding = model.encode(
            text_chunks,
            convert_to_tensor=False)
        end = time.time()
        embeddings_counter += 1

        embedding_time = round(end - start, 2)
        tqdm.write(f"⏱️ Embedding time: {embedding_time} sec")

        # Average embedding across chunks
        mean_embedding = entity_embedding.mean(axis=0).astype(float).tolist()

        embeddings_dict[dataset][dir_id] = mean_embedding
        memory_usage = print_memory()
        #embeddings_dict[dataset]['memory_GB'] = memory_usage

        # --- Periodic saving ---
        if len(embeddings_dict[dataset].keys()) % 5 == 0:
            append_to_json(file_path=output_name, new_data=embeddings_dict, dataset=dataset)
            tqdm.write(f"💾 Saved progress after {len(embeddings_dict[dataset])} embeddings.")
            # small sleep to prevent I/O overload
            if embeddings_counter % 200 == 0:
                tqdm.write(f"💾 Saved 200 embeddiings, pausing")
                time.sleep(10)
            else:
                time.sleep(0.5)
            embeddings_dict[dataset] = {}

        if embeddings_counter == THRESHOLD:
            sys.exit(f"{THRESHOLD} embeddings generated, stopping the code and then restarting")
            

    # --- Final save ---
    if embeddings_dict[dataset]:
        append_to_json(file_path=output_name, new_data=embeddings_dict, dataset=dataset)
        tqdm.write("✅ Final save completed.")

    return embeddings_dict

# --- Load Data ---
# Load the cleaned pitchbook and position files

if not os.path.exists(EMBEDDINGS_FILES_FOLDER):
    os.makedirs(EMBEDDINGS_FILES_FOLDER)

with open(EMBEDDINGS_FILES_LEGEND, 'r') as file:
    columns_mappings_legend = json.load(file)

columns_mappings = columns_mappings_legend.get(COLUMN_MAPPINGS_INDEX)
pitchbook_mappings = columns_mappings.get('pitchbook')
position_mappings = columns_mappings.get('position')

cleaned_pitchbook = pd.read_parquet(f"{PROCESSED_FOLDER}/pitchbook_company_cleaned.parquet")
cleaned_files_names = get_all_position_files()

 # Load the sentence transformer model
TRUST_REMOTE_CODE = MODEL_NAME in [
        "Lajavaness/bilingual-embedding-small",
        "Alibaba-NLP/gte-multilingual-base"
    ]
model = SentenceTransformer(model_name_or_path=MODEL_NAME, trust_remote_code=TRUST_REMOTE_CODE)
tokenizer = model.tokenizer
max_length = tokenizer.model_max_length

# Dictionary to store all embeddings
all_embeddings = dict()
all_embeddings['pitchbook'] = dict()
all_embeddings['position'] = dict()

pitchbook_ids = cleaned_pitchbook.companyid.to_list()

# Generating embeddings for pitchbook
tqdm.write('Generating embedding inputs for pitchbook')
pitchbook_texts = generate_embedding_inputs(columns_mappings=pitchbook_mappings, data=cleaned_pitchbook)

all_embeddings = generate_embeddings(embeddings_dict=all_embeddings,
                                            dataset='pitchbook',
                                            dir_list=pitchbook_ids,
                                            texts_list=pitchbook_texts,
                                            model=model,
                                            model_name=MODEL_NAME,
                                            file_name=f"embeddings_{COLUMN_MAPPINGS_INDEX}")

shift = 0

tqdm.write('Iterating over positions files:')

for position_file_name in tqdm(cleaned_files_names):

    if 'csv' in position_file_name:
        cleaned_file = pd.read_csv(f'{PROCESSED_FOLDER}/{position_file_name}')
    elif ".parquet" in position_file_name:
        cleaned_file = pd.read_parquet(f'{PROCESSED_FOLDER}/{position_file_name}')
    else:
        cleaned_file = pd.DataFrame()

    if cleaned_file.shape[0] > 0:
        
        cleaned_file.index = cleaned_file.index + shift
        cleaned_file['id'] = cleaned_file.apply(lambda x : str(x['rcid']) + '_' + str(x.name), axis = 1)

        # Extract positions company ids
        positions_ids_first = cleaned_file.id.to_list()
        positions_ids = [str(id) for id in positions_ids_first]
        
        tqdm.write('Generating embedding inputs for position')
        position_texts = generate_embedding_inputs(columns_mappings=position_mappings, data=cleaned_file, shift=shift)

        # --- Generate Embeddings for positions file --
        
        # Encode position files
        all_embeddings = generate_embeddings(embeddings_dict=all_embeddings,
                                            dataset='position',
                                            dir_list=positions_ids,
                                            texts_list=position_texts,
                                            model=model,
                                            model_name=MODEL_NAME,
                                            file_name=f"embeddings_{COLUMN_MAPPINGS_INDEX}")
        
        shift += cleaned_file.shape[0]
        del cleaned_file

