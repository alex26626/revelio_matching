# Settings

First install all the packages in requirements.txt:

pip install -r requirements.txt

In the folder alex_code you can find 4 different files, before running any of them, please modify the ROOT variable which specifies the root directory in all the files.
Ideally, this should suffice, otherwise all the directories are specified at the beginning of each file in capital letters.

Before running the files, move to the alex_code directory:

cd alex_code

# Embeddings legend

In embeddings_files/embeddings_inputs_legend.json you can find all the features configurations to be included in the embedding inputs. Each configuration is marked by an index key (e.g. "1"), which uniquely identifies each configuration.
Each configuration contains for both pitchbook and position files the features to be included as keys, and their corresponding embedding inputs as values. 

# Complete pipeline

By running:

python3 complete_pipeline.py

It is possible to generate the embeddings and find the matches using 5 different embedding models:

[
    "all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "google/embeddinggemma-300m",
    'Lajavaness/bilingual-embedding-small',
    'intfloat/multilingual-e5-small'
]

Also, for each model the matches pipeline will be run for each features configuration in the embeddings legend file. Currently in the file there are 4 different configurations. 

If you want to run the pipeline only for a specific model and a specific configuration:

python3 complete_pipeline_single.py --model_name {model_name} --columns_mappings_index {columns_mapping_index}

This should be enough, the rest is for clarification purposes. 

# Embedding pipeline

In alex_code/embedding_pipeline.py, the embedding model and the features configuration index are provided as inputs:

python3 embedding_pipeline.py --columns_mappings_index {index} --model_name {model}

Running this will generate the embeddings for each entry in the pitchbook dataset and positions dataset, which are then stored in a json file according to the following directory:
embeddings_files/embeddings_{index}/embeddings_{index}_{model_name}.json
The file is built in the following way:

"pitchbook" : {
    "pitchbook_id_1" : "embedding_1",
    "pitchbook_id_2" : "embedding_2"
}

"position" : {
    "position_id_1" : "embedding_1",
    "position_id_2" : "embedding_2"
}

These embeddings are then used to compute the similarities in matches_pipeline.py

# Matches pipeline

matches_pipeline.py takes the json file in which embeddings are stored as an input:

python3 matches_pipeline.py --embeddings_file_dir {embeddings_file_dir}

This code computes the similarities, and matches entries of the datasets by sorting the similarities from highest to lowest. 
Then, the matches are saved in an Excel file with all the used features according to the following directory:
embeddings_files/embeddings_{index}/matches_output/{model_name}_matches.xlsx

