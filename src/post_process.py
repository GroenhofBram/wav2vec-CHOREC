from os import makedirs
import os
import pandas as pd

from confusion_matrix import create_confusion_matrix, get_binary_lists
from constants import WAV2VEC2_MODEL_NAME_FLDR
from os.path import join
from pathing import get_abs_folder_path


def main():
    # This is based on the model defined as a constant.
    # CHANGE ARGUMENT IF YOU WANT TO USE A DIFFERENT FILE AS STARTING POINT
    validation_filepath_input = get_validation_filepath("total_alldata_df.csv",
                                                        type = "input")
    validation_filepath_output = get_validation_filepath("total_alldata_df_V2.csv",
                                                         type = "output")
    validaton_filepath_freqdf = get_validation_filepath("total_alldata_df_freq.csv",
                                                        type = "freq")
    validation_filepath_tomodify = get_validation_filepath("total_alldata_df.csv",
                                                           type = "modify")
    validation_filepath_confmat = get_validation_filepath("total_alldata_df_V2_ConfMat.csv",
                                                          type = "confmat")
    validation_filepath_confmat_levens = get_validation_filepath("ConfMat.csv",
                                                          type = "confmat_levens")
    # check_filepaths(validation_filepath_input, validation_filepath_output,
    #                 validaton_filepath_freqdf, validation_filepath_tomodify,
    #                 validation_filepath_confmat, validation_filepath_confmat_levens)

    validation_df = read_validation_file(validation_filepath_input)
    validation_df_FP_only = filter_FPs(validation_df)
    print(f"Number of false positives at at start\t: {len(validation_df_FP_only)}")
    validation_df_FP_freq = get_freq_FP_prompts(validation_df_FP_only)
    export_validation_df_FP_freq(validaton_filepath_freqdf, validation_df_FP_freq)


    # Print to see overview
    # print(validation_df_FP_only)
    validation_df_working = fix_spaces(validation_df)
    # print(validation_df_working) 
    new_validation_df = create_new_validation_df(validation_df, validation_df_working)

    new_validation_df = add_error_scores(new_validation_df)
    print(new_validation_df)

    export_new_validation_df(new_validation_df, validation_filepath_output)

    ref_list_binary, hyp_list_binary = get_binary_lists(new_validation_df)

    conf_matrix = create_confusion_matrix(ref_list_binary, hyp_list_binary)
    print(f"\nNew confusion matrix values:\n{conf_matrix}")

    export_new_confmat(conf_matrix, validation_filepath_confmat)

    # I tried doing this with a generator, but didn't work. It's ugly!
    threshold_a = 0.05
    threshold_b = 0.10
    threshold_c = 0.15
    threshold_d = 0.20
    threshold_e = 0.25
    threshold_f = 0.30
    threshold_g = 0.35
    threshold_h = 0.40
    threshold_i = 0.45
    threshold_j = 0.50
    threshold_k = 0.55
    threshold_l = 0.60
    threshold_m = 0.65
    threshold_n = 0.70
    threshold_o = 0.75
    threshold_p = 0.80
    threshold_q = 0.85
    threshold_r = 0.90
    threshold_s = 0.95
    threshold_t = 1.00
    thresholds = [threshold_a, threshold_b, threshold_c, threshold_d,
                  threshold_e, threshold_f, threshold_g, threshold_h,
                  threshold_i, threshold_j, threshold_k, threshold_l,
                  threshold_m, threshold_n, threshold_o, threshold_p,
                  threshold_q, threshold_r, threshold_s, threshold_t]

    for threshold in thresholds:
        new_validation_df_levensh = calc_with_Levenshteim_distance(new_validation_df, threshold)
        create_export_new_validation_df_levensh_confmat(new_validation_df_levensh, threshold, validation_filepath_confmat_levens)





def create_export_new_validation_df_levensh_confmat(new_validation_df_levensh, threshold, validation_filepath_confmat_levens):
    ref_list_binary, hyp_list_binary = get_binary_lists(new_validation_df_levensh)
    conf_matrix = create_confusion_matrix(ref_list_binary, hyp_list_binary)
    conf_matrix_df = pd.DataFrame(conf_matrix)

    new_validation_df_levensh_filepath = validation_filepath_confmat_levens.replace("ConfMat.csv", str(threshold) + "DF.csv")
    new_validation_df_levensh.to_csv(new_validation_df_levensh_filepath, index = False)

    conf_mat_filepath = validation_filepath_confmat_levens.replace("ConfMat.csv", str(threshold) + "ConfMat.csv")    

    conf_matrix_df.to_csv(conf_mat_filepath, index = False)
    print(f"\nConfmat for threshold {threshold} stored at\t:{conf_mat_filepath}")
    print(f"\n- - - - - CONFMAT VALUES FOR THRESHOLD {threshold} - - - - -")
    print(conf_matrix)
    print("\n -----------------------------------------------------------\n")


def calc_with_Levenshteim_distance(new_validation_df, threshold):
    new_validation_df['prompts_plus_hypo'] = 1  # Set all to 1

    for index, row in new_validation_df.iterrows():
        if row['score'] >= threshold or row['score_rev'] >= threshold:
            new_validation_df.at[index, 'prompts_plus_hypo'] = 0
    return new_validation_df


def add_error_scores(new_validation_df):
    new_validation_df['score'] = 0
    new_validation_df["insertions"] = 0
    new_validation_df["deletions"] = 0
    new_validation_df["substitutions"] = 0

    new_validation_df['score_rev'] = 0
    new_validation_df["insertions_rev"] = 0
    new_validation_df["deletions_rev"] = 0
    new_validation_df["substitutions_rev"] = 0

    for index, row in new_validation_df.iterrows():
        # print("-------------")
        # print(row)
        # print("-------------")
        prompt_length = len(row['prompt_aligned'])
        prompt_length_rev = len(row['prompt_aligned_rev'])

        hypothesis = row['hypothesis']
        prompt_aligned = row['prompt_aligned']
        
        hypothesis_rev = row["hypothesis_rev"]
        prompt_aligned_rev = row["prompt_aligned_rev"]
        
        insertions = sum(1 for h, p in zip(hypothesis, prompt_aligned) if h != '*' and p == '*')
        insertions_rev = sum(1 for h, p in zip(hypothesis_rev, prompt_aligned_rev) if h != '*' and p == '*')

        deletions = sum(1 for h, p in zip(hypothesis, prompt_aligned) if h == '*' and p != '*')
        deletions_rev = sum(1 for h, p in zip(hypothesis_rev, prompt_aligned_rev) if h == '*' and p != '*')

        substitutions = sum(1 for h, p in zip(hypothesis, prompt_aligned) if h == '*' and p == '*')
        substitutions_rev = sum(1 for h, p in zip(hypothesis_rev, prompt_aligned_rev) if h == '*' and p == '*')
        
        new_validation_df.at[index, 'insertions'] = insertions
        new_validation_df.at[index, 'insertions_rev'] = insertions_rev

        new_validation_df.at[index, 'deletions'] = deletions    
        new_validation_df.at[index, 'deletions_rev'] = deletions_rev

        new_validation_df.at[index, 'substitutions'] = substitutions
        new_validation_df.at[index, 'substitutions_rev'] = substitutions_rev

        new_validation_df.at[index, 'score'] = round( (prompt_length - (insertions + deletions + substitutions) )/(prompt_length), 2)
        new_validation_df.at[index, 'score_rev'] = round( (prompt_length_rev - (insertions_rev + deletions_rev + substitutions_rev) )/(prompt_length_rev), 2)

    return new_validation_df

def export_new_confmat(conf_matrix, validation_filepath_confmat: str):
    conf_matrix_df = pd.DataFrame(conf_matrix)

    conf_matrix_df.to_csv(validation_filepath_confmat, index = False)
    # print(f"\nNew Confmat stored at\t:{validation_filepath_confmat}")

def export_new_validation_df(new_validation_df, validation_filepath_output):
    new_validation_df.to_csv(validation_filepath_output, index=False)
    # print(f"\nNew Validation set stored at\t:{validation_filepath_output}")

    
def create_new_validation_df(validation_df, validation_df_working):
    combined_df = pd.concat([validation_df, validation_df_working])
    new_validation_df = combined_df.drop_duplicates(subset=['id', 'prompt'], keep='last')
    
    return new_validation_df


def fix_spaces(validation_df_FP_only):
    for index, row in validation_df_FP_only.iterrows():
        prompt = row['prompt'].replace(" ", "")
        hypothesis = row['hypothesis'].replace(" ", "")
        hypothesis_rev = row['hypothesis_rev'].replace(" ", "")
        
        if prompt == hypothesis or prompt == hypothesis_rev:
            validation_df_FP_only.at[index, 'prompts_plus_hypo'] = 0

    return validation_df_FP_only


def export_validation_df_FP_freq(filepath: str, freq_df):
    freq_df.to_csv(filepath, index=False)



def get_validation_filepath(file_name: str, type: str):
    base_output_dir_in_repo = get_abs_folder_path("output")
    csv_dir = join(base_output_dir_in_repo, WAV2VEC2_MODEL_NAME_FLDR)
    csv_dir_input = join(csv_dir, "all_data_output")
    csv_dir = join(csv_dir_input, "post_processing")
    makedirs(csv_dir, exist_ok=True)
    csv_dir_confmat_levens = join(csv_dir, "by_levens_threshold")
    makedirs(csv_dir_confmat_levens, exist_ok=True) 

    if type == "input":
        csv_input_filepath = join(csv_dir_input, file_name)
        return csv_input_filepath
    
    elif type == "output":
        csv_output_filepath = join(csv_dir, file_name)
        return csv_output_filepath
    
    elif type == "freq":
        csv_freq_output_filepath = join(csv_dir, file_name)
        return csv_freq_output_filepath
    
    elif type == "modify":
        csv_to_modify_filepath = join(csv_dir_input, file_name)
        return csv_to_modify_filepath

    elif type == "confmat":
        csv_output_filepath_confmat = join(csv_dir, file_name)
        return csv_output_filepath_confmat
    
    elif type == "confmat_levens":
        csv_output_filepath_confmat_levens = join(csv_dir_confmat_levens, file_name)
        return csv_output_filepath_confmat_levens


def check_filepaths(s1: str, s2: str, s3: str, s4: str, s5: str, s6: str):
    print(f"\n+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ \nPlease check the filepaths:\nInput\t:{s1}\nOutput\t:{s2}\nFreq\t:{s3}\nMod\t:{s4}\nConfMat\t:{s5}\nConfMat_Levens\t:{s6}\n+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")
    input("\nPress any key to continue...")

def read_validation_file(validation_filepath: str):
    df = pd.read_csv(validation_filepath)

    return df

def filter_FPs(validation_df):
    filtered_df = validation_df[(validation_df['prompts_plus_orth'] == 0) &
                                (validation_df['prompts_plus_hypo'] == 1)]
    return filtered_df

def get_freq_FP_prompts(validation_df_FP_only):
    freq_df = validation_df_FP_only['prompt'].value_counts().reset_index()
    
    return freq_df