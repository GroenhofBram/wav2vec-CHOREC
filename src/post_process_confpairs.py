from os import makedirs
import os
import pandas as pd

from confusion_matrix import create_confusion_matrix, get_binary_lists
from constants import WAV2VEC2_MODEL_NAME_FLDR
from os.path import join
from pathing import get_abs_folder_path


def main():
    # # # Change these based on which file to use and if there were manual rules edited in post_process., also bear model name in mind
    confpair_output_dir, confpair_output_csv_path = get_confpair_filepaths(csv_name = "full_set_base.csv")
    data, data_input_file = get_all_data(man_edit = False)
    # # # 



    print("\n---------------------------------------------------------\n")
    print(f"Data taken from this file\t: {data_input_file}")
    print(f"Confusion pairs directory\t: {confpair_output_dir}")
    print(f"Confusion pairs file found\t: {confpair_output_csv_path}")
    print("\n---------------------------------------------------------\n")
    input("\nIf correct, press any key to continue...")

    data = fix_prompts(data)
    data_confpairs_only = get_confpairs_only(data)

    create_confpairs_freq_file(data_confpairs_only, confpair_output_csv_path)

def create_confpairs_freq_file(data_confpairs_only, confpair_output_csv_path):
    filtered_data = data_confpairs_only[['prompt', 'hypothesis', 'prompt_aligned']]
    filtered_data['count'] = 1

    confusion_pairs_freq_df = filtered_data.groupby(['prompt', 'hypothesis'], as_index=False).sum()
   
    confusion_pairs_freq_df = confusion_pairs_freq_df.drop(columns=['prompt_aligned'])
    distinct_prompt_aligned = filtered_data.drop_duplicates(subset=['prompt', 'hypothesis'])[['prompt', 'hypothesis', 'prompt_aligned']]
    confusion_pairs_freq_df = confusion_pairs_freq_df.merge(distinct_prompt_aligned, on=['prompt', 'hypothesis'], how='left')
    


    # confusion_pairs_freq_df = filtered_data.groupby(['prompt', 'hypothesis'], as_index=False).sum()
    confusion_pairs_freq_df.to_csv(confpair_output_csv_path)
    print(f"Confpairs stored at\t: {confpair_output_csv_path}")
    

def get_confpairs_only(data):
    confpairs_only = data[(data['prompts_plus_orth'] == 0) & (data['prompts_plus_hypo'] == 1)]
    return confpairs_only

def fix_prompts(data):
    data.loc[data['reference'] == 'CORRECTLY_READ', 'reference'] = data['prompt']+"<CORRECT>"
    return data


def get_all_data(man_edit: bool):
    base_output_dir_in_repo = get_abs_folder_path("output")
    csv_dir = join(base_output_dir_in_repo, WAV2VEC2_MODEL_NAME_FLDR)
    csv_data_folder = join(csv_dir, "all_data_output")

    if man_edit == True:
        csv_data_file = join(csv_data_folder, "post_processing")
        csv_data_file = join(csv_data_file, "total_alldata_df_V2.csv")
    else:
        csv_data_file = join(csv_data_folder, "total_alldata_df.csv")

    data = pd.read_csv(csv_data_file)

    return data, csv_data_file




def get_confpair_filepaths(csv_name: str):
    base_output_dir_in_repo = get_abs_folder_path("output")
    csv_dir = join(base_output_dir_in_repo, WAV2VEC2_MODEL_NAME_FLDR)
    csv_dir_input = join(csv_dir, "all_data_output")
    csv_dir = join(csv_dir_input, "post_processing")
    csv_dir = join(csv_dir, "confusion_pairs")
    makedirs(csv_dir, exist_ok=True)

    csv_file_name = join(csv_dir, csv_name)

    return csv_dir, csv_file_name