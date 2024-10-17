import pandas as pd
import re
from constants import WAV2VEC2_MODEL_NAME_FLDR
from os.path import join
from pathing import get_abs_folder_path

def main():
    # Preparing files
    filepath, filepath_output = get_filename(csv_name = "full_set_base.csv")
    
    print(f"File will be read from\t: {filepath}")    
    print(f"File will be stored in\t: {filepath_output}")
    input("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\tIS THIS CORRECT? PRESS ANY KEY TO CONTINUE\n")
    
    data = read_error_csv(filepath)
    error_type_df = create_empty_error_type_df()

    # FINDING DIFFERENT TYPES OF ERRORS #
    error_type_df = add_nospace(data, error_type_df)

    error_type_df = add_oe_oo(data, error_type_df)

    error_type_df = add_k_c(data, error_type_df)

    error_type_df = add_long_short_vowel(data, error_type_df) 

    error_type_df = add_au_ou(data, error_type_df)

    error_type_df = add_start_plosive(data, error_type_df)
    error_type_df = add_end_plosive(data, error_type_df)
    error_type_df = add_other_plosive(data, error_type_df)

    error_type_df = add_double_single_consonant(data, error_type_df, asterisk = True)
    error_type_df = add_double_single_consonant(data, error_type_df, asterisk = False)

    error_type_df = add_i_replace(data, error_type_df)

    error_type_df = add_ch_confusion(data, error_type_df)

    error_type_df = add_start_fricative(data, error_type_df)
    error_type_df = add_other_fricative(data, error_type_df)

    error_type_df = add_nasal(data, error_type_df)

    error_type_df = add_final_letter_deletion(data, error_type_df)

    error_type_df = add_liquid_deletion(data, error_type_df)

    error_type_df = add_insertions(data, error_type_df)

    # # # # # #




    print(error_type_df)
    print(sum(error_type_df["count"]))
    # print(error_type_df[error_type_df["prompt"] == "konijnenhok"])

    save_to_csv(error_type_df, filepath_output)



def save_to_csv(error_type_df, filepath_output):
    error_type_df.to_csv(filepath_output)

def add_liquid_deletion(data, error_type_df):
    rows_to_add = []

    for i, row in data.iterrows():
        if differ_by_liquid_deletion(row['prompt'], row['hypothesis']):
            if not row_exists(error_type_df, row['prompt'], row['hypothesis'], row['count']):
                new_row = {
                    'error_type': 'liquid_deletion',
                    'prompt': row['prompt'],
                    'hypothesis': row['hypothesis'],
                    'count': row['count'],
                    'prompt_aligned': row['prompt_aligned']
                }

                rows_to_add.append(new_row)


    if rows_to_add:
        error_type_df = pd.concat([error_type_df, pd.DataFrame(rows_to_add)], ignore_index=True)
    
    return error_type_df

def add_insertions(data, error_type_df):
    rows_to_add = []

    for i, row in data.iterrows():
        if differ_by_insertion(row['prompt'], row['hypothesis']):
            if not row_exists(error_type_df, row['prompt'], row['hypothesis'], row['count']):
                new_row = {
                    'error_type': 'insertion',
                    'prompt': row['prompt'],
                    'hypothesis': row['hypothesis'],
                    'count': row['count'],
                    'prompt_aligned': row['prompt_aligned']
                }
                rows_to_add.append(new_row)

    if rows_to_add:
        error_type_df = pd.concat([error_type_df, pd.DataFrame(rows_to_add)], ignore_index=True)
    
    return error_type_df

def differ_by_insertion(prompt, hypothesis):
    # Check if hypothesis is longer than prompt by exactly one character
    if len(hypothesis) != len(prompt) + 1:
        return False

    # Find the length of prompt and hypothesis
    len_prompt = len(prompt)
    len_hypothesis = len(hypothesis)

    # Iterate through the characters of both strings
    for i in range(len_hypothesis):
        if i < len_prompt:
            if prompt[i] != hypothesis[i]:
                # If a mismatch is found, check if it's due to an insertion
                if prompt[i] != hypothesis[i + 1]:
                    return False
                else:
                    return hypothesis.startswith(prompt[:i]) and hypothesis[i + 1:] == prompt[i:]
    
    # If all characters are the same up to the last one, it's an insertion
    return hypothesis.startswith(prompt)

def differ_by_liquid_deletion(prompt, hypothesis):
    liquid_patterns = [r'([bcdfghjklmnpqrstvwxyz])([rl])', r'([aeiou])([rl])']


    modified_prompt = prompt
    modified_hypothesis = hypothesis
    
    for pattern in liquid_patterns:
        modified_prompt = re.sub(pattern, r'\1*', modified_prompt)
        modified_hypothesis = re.sub(pattern, r'\1*', modified_hypothesis)

    return modified_prompt == hypothesis or modified_hypothesis == prompt

def add_final_letter_deletion(data, error_type_df):
    rows_to_add = []

    for i, row in data.iterrows():
        if differ_by_final_letter(row['prompt'], row['hypothesis']):
            if not row_exists(error_type_df, row['prompt'], row['hypothesis'], row['count']):
                new_row = {
                    'error_type': 'final_deletion',
                    'prompt': row['prompt'],
                    'hypothesis': row['hypothesis'],
                    'count': row['count'],
                    'prompt_aligned': row['prompt_aligned']
                }
                rows_to_add.append(new_row)

    if rows_to_add:
        error_type_df = pd.concat([error_type_df, pd.DataFrame(rows_to_add)], ignore_index=True)
    
    return error_type_df

def differ_by_final_letter(prompt, hypothesis): 
    if len(prompt) > 1 and len(hypothesis) > 1:
        if prompt[:-1] == hypothesis[:-1]:
            if prompt[-1] == '*' or hypothesis[-1] == '*' or prompt[-1] == hypothesis[-1]:
                return True
    elif len(prompt) == 1 and len(hypothesis) == 1:
        if prompt == '*' or hypothesis == '*' or prompt == hypothesis:
            return True
    return False    

def add_nasal(data, error_type_df):
    rows_to_add = []


    for i, row in data.iterrows():
        if differ_by_nasal(row['prompt'], row['hypothesis']):
            if not row_exists(error_type_df, row['prompt'], row['hypothesis'], row['count']):
                new_row = {
                    'error_type': 'nasal',
                    'prompt': row['prompt'],
                    'hypothesis': row['hypothesis'],
                    'count': row['count'],
                    'prompt_aligned': row['prompt_aligned']
                }
                rows_to_add.append(new_row)

    if rows_to_add:
        error_type_df = pd.concat([error_type_df, pd.DataFrame(rows_to_add)], ignore_index=True)
    
    return error_type_df

def differ_by_nasal(prompt, hypothesis):
    nasal_pairs = [('n', 'm'), ('m', 'n'), 
                    ('n', 'ng'), ('ng', 'n'), 
                    ('m', 'ng'), ('ng', 'm'),
                    ('n', 'nk'), ('nk', 'n'), 
                    ('m', 'nk'), ('nk', 'm'), 
                    ('ng', 'nk'), ('nk', 'ng'),
                    ('n', 'm'), ('n', 'ng'), 
                    ('n', 'nk'), ('ng', 'n'), 
                    ('ng', 'm'), ('ng', 'nk'), 
                    ('m', 'n'), ('m', 'ng'), 
                    ('m', 'nk'), ('nk', 'n'), 
                    ('nk', 'ng'), ('nk', 'm'),
                    ('n', '*k'), ('n', 'n*'), 
                    ('ng', '*k'), ('ng', 'n*'), 
                    ('m', '*k'), ('m', 'n*'), 
                    ('nk', '*k'), ('nk', 'n*')]
    
    for p1, p2 in nasal_pairs:
        if (p1 in prompt and p2 in hypothesis) or (p1 in hypothesis and p2 in prompt):
            transformed_prompt = prompt.replace(p1, p2)
            transformed_hypothesis = hypothesis.replace(p1, p2)
            if transformed_prompt == transformed_hypothesis:
                return True
            
    return False

def add_other_fricative(data, error_type_df):
    rows_to_add = []


    for i, row in data.iterrows():
        if differ_by_start_fricative(row['prompt'], row['hypothesis']):
            if not row_exists(error_type_df, row['prompt'], row['hypothesis'], row['count']):
                new_row = {
                    'error_type': 'other_fricative',
                    'prompt': row['prompt'],
                    'hypothesis': row['hypothesis'],
                    'count': row['count'],
                    'prompt_aligned': row['prompt_aligned']
                }
                rows_to_add.append(new_row)

    if rows_to_add:
        error_type_df = pd.concat([error_type_df, pd.DataFrame(rows_to_add)], ignore_index=True)
    
    return error_type_df    

def differ_by_other_fricative(prompt, hypothesis):
    fricative_pairs = [("f", "v"), ("s", "z"), ("g", "ch")]

    for i in range(len(prompt)):
        for p1, p2 in fricative_pairs:
            if ((prompt[i] == p1 and hypothesis[i] == p2) or (prompt[i] == p2 and hypothesis[i] == p1)) and \
               (prompt[:i] + prompt[i+1:] == hypothesis[:i] + hypothesis[i+1:]):
                return True
                
    return False

def add_start_fricative(data, error_type_df):
    rows_to_add = []


    for i, row in data.iterrows():
        if differ_by_start_fricative(row['prompt'], row['hypothesis']):
            if not row_exists(error_type_df, row['prompt'], row['hypothesis'], row['count']):
                new_row = {
                    'error_type': 'start_fricative',
                    'prompt': row['prompt'],
                    'hypothesis': row['hypothesis'],
                    'count': row['count'],
                    'prompt_aligned': row['prompt_aligned']
                }
                rows_to_add.append(new_row)

    if rows_to_add:
        error_type_df = pd.concat([error_type_df, pd.DataFrame(rows_to_add)], ignore_index=True)
    
    return error_type_df

def differ_by_start_fricative(prompt, hypothesis):
    fricative_pairs = [("f", "v"), ("s", "z"), ("g", "ch")]
    
    for p1, p2 in fricative_pairs:
        if (prompt.startswith(p1) and hypothesis.startswith(p2) and prompt[1:] == hypothesis[1:]) or \
           (prompt.startswith(p2) and hypothesis.startswith(p1) and prompt[1:] == hypothesis[1:]):
            return True
    return False

def add_ch_confusion(data, error_type_df):
    rows_to_add = []

    for i, row in data.iterrows():
        if differ_by_ch_confusion(row['prompt'], row['hypothesis']):
            if not row_exists(error_type_df, row['prompt'], row['hypothesis'], row['count']):            
                new_row = {
                    'error_type': 'ch confusion',
                    'prompt': row['prompt'],
                    'hypothesis': row['hypothesis'],
                    'count': row['count'],
                    'prompt_aligned': row['prompt_aligned']
                }
                rows_to_add.append(new_row)

    if rows_to_add:
        error_type_df = pd.concat([error_type_df, pd.DataFrame(rows_to_add)], ignore_index=True)
    return error_type_df

def differ_by_ch_confusion(prompt, hypothesis):
    len_prompt = len(prompt)
    len_hypothesis = len(hypothesis)
    

    
    # Check for "ch" in prompt and its absence in hypothesis at the same positions
    for i in range(len_prompt - 1):
        if prompt[i:i+2] == 'ch':
            if hypothesis[i:i+2] != 'ch':
                return True
    
    for i in range(len_hypothesis - 1):
        if hypothesis[i:i+2] == 'ch':
            if prompt[i:i+2] != 'ch':
                return True
    
    return False

def add_i_replace(data, error_type_df):
    rows_to_add = []

    for i, row in data.iterrows():
        if differ_by_i(row['prompt'], row['hypothesis']):
            if not row_exists(error_type_df, row['prompt'], row['hypothesis'], row['count']):
                new_row = {
                    'error_type': 'i_replace',
                    'prompt': row['prompt'],
                    'hypothesis': row['hypothesis'],
                    'count': row['count'],
                    'prompt_aligned': row['prompt_aligned']
                }
                rows_to_add.append(new_row)

    if rows_to_add:
        error_type_df = pd.concat([error_type_df, pd.DataFrame(rows_to_add)], ignore_index=True)
    return error_type_df

def differ_by_i(prompt, hypothesis):
    condition1 = prompt.replace("i", "y") == hypothesis
    condition2 = prompt.replace("y", "i") == hypothesis
    return condition1 or condition2 

def add_double_single_consonant(data, error_type_df, asterisk):
    rows_to_add = []

    if asterisk:
        for i, row in data.iterrows():
            if differ_by_double_single_consonant(row['prompt'], row['hypothesis']):
                if not row_exists(error_type_df, row['prompt'], row['hypothesis'], row['count']):
                    new_row = {
                        'error_type': 'double/single consonant',
                        'prompt': row['prompt'],
                        'hypothesis': row['hypothesis'],
                        'count': row['count'],
                        'prompt_aligned': row['prompt_aligned']
                    }
                    rows_to_add.append(new_row)
    
    else:
         for i, row in data.iterrows():
            if differ_by_double_single_consonant_no_asterisk(row['prompt'], row['hypothesis']):
                if not row_exists(error_type_df, row['prompt'], row['hypothesis'], row['count']):
                    new_row = {
                        'error_type': 'double/single consonant',
                        'prompt': row['prompt'],
                        'hypothesis': row['hypothesis'],
                        'count': row['count'],
                        'prompt_aligned': row['prompt_aligned']
                    }
                    rows_to_add.append(new_row)       

    if rows_to_add:
        error_type_df = pd.concat([error_type_df, pd.DataFrame(rows_to_add)], ignore_index=True)

    return error_type_df

def differ_by_double_single_consonant_no_asterisk(prompt, hypothesis):
    pattern = r'([bcdfghjklmnpqrstvwxyz])\1|\*([bcdfghjklmnpqrstvwxyz])'
    base_prompt = re.sub(pattern, r'\1', prompt)
    base_hypothesis = re.sub(pattern, r'\1', hypothesis)
    
    return base_prompt == base_hypothesis    

def differ_by_double_single_consonant(prompt, hypothesis):
    pattern_double_to_single = r'([bcdfghjklmnpqrstvwxyz])\1'  # Matches double consonants
    pattern_single_to_double = r'\*([bcdfghjklmnpqrstvwxyz])'  # Matches "* followed by single consonant"
    

    base_prompt_double_to_single = re.sub(pattern_double_to_single, r'*\1', prompt)
    base_prompt_single_to_double = re.sub(pattern_single_to_double, r'\1\1', prompt)
    
    base_hypothesis_double_to_single = re.sub(pattern_double_to_single, r'*\1', hypothesis)
    base_hypothesis_single_to_double = re.sub(pattern_single_to_double, r'\1\1', hypothesis)
    
    return (base_prompt_double_to_single == base_hypothesis_double_to_single) or \
           (base_prompt_single_to_double == base_hypothesis_single_to_double)

def add_other_plosive(data, error_type_df):
    rows_to_add = []

    for i, row in data.iterrows():
        if differ_by_other_plosive(row['prompt'], row['hypothesis']):
            if not row_exists(error_type_df, row['prompt'], row['hypothesis'], row['count']):
                new_row = {
                    'error_type': 'other_plosive',
                    'prompt': row['prompt'],
                    'hypothesis': row['hypothesis'],
                    'count': row['count'],
                    'prompt_aligned': row['prompt_aligned']
                }

                rows_to_add.append(new_row)


    if rows_to_add:
        error_type_df = pd.concat([error_type_df, pd.DataFrame(rows_to_add)], ignore_index=True)
    
    return error_type_df

def differ_by_other_plosive(prompt, hypothesis):
    plosive_pairs = [("p", "b"), ("t", "d")]

    # Ensure prompt and hypothesis are of the same length
    if len(prompt) != len(hypothesis):
        return False

    # Flag to track if a mismatch other than plosive substitution is found
    mismatch_found = False

    for i in range(len(prompt)):
        if prompt[i] != hypothesis[i]:
            # Check if the characters are not the same, and if they are plosive pairs
            if (prompt[i], hypothesis[i]) not in plosive_pairs and (hypothesis[i], prompt[i]) not in plosive_pairs:
                mismatch_found = True
                break

    # If no other mismatches were found and there was at least one plosive substitution
    return not mismatch_found

def add_end_plosive(data, error_type_df):
    rows_to_add = []

    for i, row in data.iterrows():
        if differ_by_end_plosive(row['prompt'], row['hypothesis']):
            if not row_exists(error_type_df, row['prompt'], row['hypothesis'], row['count']):
                new_row = {
                    'error_type': 'end_plosive',
                    'prompt': row['prompt'],
                    'hypothesis': row['hypothesis'],
                    'count': row['count'],
                    'prompt_aligned': row['prompt_aligned']
                }
                rows_to_add.append(new_row)

    if rows_to_add:
        error_type_df = pd.concat([error_type_df, pd.DataFrame(rows_to_add)], ignore_index=True)
    
    return error_type_df

def differ_by_end_plosive(prompt, hypothesis):
    plosive_pairs = [("p", "b"), ("t", "d"), ("k", "g")]
    
    for p1, p2 in plosive_pairs:
        if (prompt.endswith(p1) and hypothesis.endswith(p2) and prompt[:-1] == hypothesis[:-1]) or \
           (prompt.endswith(p2) and hypothesis.endswith(p1) and prompt[:-1] == hypothesis[:-1]):
            return True
    return False

def add_start_plosive(data, error_type_df):
    rows_to_add = []


    for i, row in data.iterrows():
        if differ_by_start_plosive(row['prompt'], row['hypothesis']):
            if not row_exists(error_type_df, row['prompt'], row['hypothesis'], row['count']):
                new_row = {
                    'error_type': 'start_plosive',
                    'prompt': row['prompt'],
                    'hypothesis': row['hypothesis'],
                    'count': row['count'],
                    'prompt_aligned': row['prompt_aligned']
                }
                rows_to_add.append(new_row)

    if rows_to_add:
        error_type_df = pd.concat([error_type_df, pd.DataFrame(rows_to_add)], ignore_index=True)
    
    return error_type_df

def differ_by_start_plosive(prompt, hypothesis):
    plosive_pairs = [("p", "b"), ("t", "d"), ("k", "g")]
    
    for p1, p2 in plosive_pairs:
        if (prompt.startswith(p1) and hypothesis.startswith(p2) and prompt[1:] == hypothesis[1:]) or \
           (prompt.startswith(p2) and hypothesis.startswith(p1) and prompt[1:] == hypothesis[1:]):
            return True
    return False

def add_long_short_vowel(data, error_type_df):
    rows_to_add = []

    for i, row in data.iterrows():
        if differ_by_long_short_vowel(row['prompt'], row['hypothesis']):
            if not row_exists(error_type_df, row['prompt'], row['hypothesis'], row['count']):
                new_row = {
                    'error_type': 'long/short vowel',
                    'prompt': row['prompt'],
                    'hypothesis': row['hypothesis'],
                    'count': row['count'],
                    'prompt_aligned': row['prompt_aligned']
                }
                rows_to_add.append(new_row)

    if rows_to_add:
        error_type_df = pd.concat([error_type_df, pd.DataFrame(rows_to_add)], ignore_index=True)
    
    return error_type_df

def differ_by_long_short_vowel(prompt, hypothesis):
    long_vowels = ["aa", "ee", "oo", "uu", "ii",
                   "ei", "ij", "ie", "oe", 
                   "au", "ou", "eu",
                   "aai", "ooi", "eeu"]
    
    for long_vowel in long_vowels:
        # Check if long vowel in prompt is replaced by a single vowel or a vowel with a "*"
        if long_vowel in prompt:
            single_vowel = long_vowel[0]
            if (prompt.replace(long_vowel, single_vowel) == hypothesis or 
                prompt.replace(long_vowel, f"{single_vowel}*") == hypothesis or
                prompt.replace(long_vowel, f"*{single_vowel}") == hypothesis):
                return True
        # Check if single vowel or vowel with "*" in prompt is replaced by long vowel
        if long_vowel in hypothesis:
            single_vowel = long_vowel[0]
            if (hypothesis.replace(long_vowel, single_vowel) == prompt or 
                hypothesis.replace(long_vowel, f"{single_vowel}*") == prompt or
                hypothesis.replace(long_vowel, f"*{single_vowel}") == hypothesis):
                return True
    return False

def add_k_c(data, error_type_df):
    rows_to_add = []

    for i, row in data.iterrows():
        if differ_by_k_c(row['prompt'], row['hypothesis']):
            if not row_exists(error_type_df, row['prompt'], row['hypothesis'], row['count']):
                new_row = {
                    'error_type': 'k/c',
                    'prompt': row['prompt'],
                    'hypothesis': row['hypothesis'],
                    'count': row['count'],
                    'prompt_aligned': row['prompt_aligned']
                }
                rows_to_add.append(new_row)

    if rows_to_add:
        error_type_df = pd.concat([error_type_df, pd.DataFrame(rows_to_add)], ignore_index=True)
    return error_type_df

def differ_by_k_c(prompt, hypothesis):
    condition1 = prompt.replace("k", "c") == hypothesis
    condition2 = prompt.replace("c", "k") == hypothesis
    return condition1 or condition2

def add_oe_oo(data, error_type_df):
    rows_to_add = []
    for i, row in data.iterrows():
        if differ_by_oe_oo(row['prompt'], row['hypothesis']):
            if not row_exists(error_type_df, row['prompt'], row['hypothesis'], row['count']):
                new_row = {
                    'error_type': 'oe/oo',
                    'prompt': row['prompt'],
                    'hypothesis': row['hypothesis'],
                    'count': row['count'],
                    'prompt_aligned': row['prompt_aligned']
                }
                rows_to_add.append(new_row)

    
    if rows_to_add:
        error_type_df = pd.concat([error_type_df, pd.DataFrame(rows_to_add)], ignore_index=True)
    
    return error_type_df

def differ_by_oe_oo(prompt, hypothesis):
    condition1 = prompt.replace("oe", "oo") == hypothesis
    condition2 = prompt.replace("oo", "oe") == hypothesis
    return condition1 or condition2

def add_au_ou(data, error_type_df):
    rows_to_add = []

    for i, row in data.iterrows():
        if differ_by_au_ou(row['prompt'], row['hypothesis']):
            if not row_exists(error_type_df, row['prompt'], row['hypothesis'], row['count']):
            
                new_row = {
                    'error_type': 'au/ou',
                    'prompt': row['prompt'],
                    'hypothesis': row['hypothesis'],
                    'count': row['count'],
                    'prompt_aligned': row['prompt_aligned']
                }
                
                rows_to_add.append(new_row)

    if rows_to_add:
        error_type_df = pd.concat([error_type_df, pd.DataFrame(rows_to_add)], ignore_index=True)
    
    return error_type_df

def differ_by_au_ou(prompt, hypothesis):
    condition1 = prompt.replace("au", "ou") == hypothesis
    condition2 = prompt.replace("ou", "au") == hypothesis

    return condition1 or condition2

def add_nospace(data, error_type_df):
    rows_to_add = []

    for i, row in data.iterrows():
        if check_nospace_conditions(row['prompt'], row['hypothesis']):
            if not row_exists(error_type_df, row['prompt'], row['hypothesis'], row['count']):
                new_row = {
                    'error_type': 'nospace',
                    'prompt': row['prompt'],
                    'hypothesis': row['hypothesis'],
                    'count': row['count'],
                    'prompt_aligned': row['prompt_aligned']
                }
                rows_to_add.append(new_row)

    if rows_to_add:
        error_type_df = pd.concat([error_type_df, pd.DataFrame(rows_to_add)], ignore_index=True)
    
    return error_type_df

def check_nospace_conditions(prompt, hypothesis):
    hypothesis_no_space = hypothesis.replace(" ", "")
    same_without_spaces = hypothesis_no_space == prompt
    no_asterisk = "*" not in hypothesis
    return same_without_spaces and no_asterisk

def row_exists(df, prompt, hypothesis, count):
    return ((df['prompt'] == prompt) & (df['hypothesis'] == hypothesis) & (df['count'] == count)).any()

def create_empty_error_type_df():
    columns = ['error_type', 'prompt', 'hypothesis', 'count', 'prompt_aligned']
    empty_df = pd.DataFrame(columns=columns)
    return empty_df

def read_error_csv(filepath: str):
    data = pd.read_csv(filepath)
    return data

def get_filename(csv_name: str):
    base_output_dir_in_repo = get_abs_folder_path("output")
    csv_dir = join(base_output_dir_in_repo, WAV2VEC2_MODEL_NAME_FLDR)
    csv_dir = join(csv_dir, "all_data_output")
    csv_dir = join(csv_dir, "post_processing")
    csv_dir = join(csv_dir, "confusion_pairs")
    csv_file_name = join(csv_dir, csv_name)
    csv_file_name_output = join(csv_dir, "ERR_TYPES-" + csv_name)

    return csv_file_name, csv_file_name_output