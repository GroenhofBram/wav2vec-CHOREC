import os

dir_path = "D:/repos/wav2vec-CHOREC/output/GroNLP-FULL-wav2vec2-dutch-large-ft-cgn" 
          #"D:/repos/whisper-CHOREC/output/GroNLP-TRAINING-WHISPERwav2vec2-dutch-large-ft-cgn"


def count_wordlists(dir_path):
    total_dirs = 0
    count_1LG = 0
    count_2LG = 0
    count_3_4LG = 0
    
    # List directories in the specified directory
    for d in os.listdir(dir_path):
        full_path = os.path.join(dir_path, d)
        if os.path.isdir(full_path):
            total_dirs += 1
            if d.endswith("1LG"):
                count_1LG += 1
            elif d.endswith("2LG"):
                count_2LG += 1
            elif d.endswith("3+4LG"):
                count_3_4LG += 1
    
    # Calculate proportions
    if total_dirs == 0:
        proportion_1LG = 0.0
        proportion_2LG = 0.0
        proportion_3_4LG = 0.0
    else:
        proportion_1LG = count_1LG / total_dirs
        proportion_2LG = count_2LG / total_dirs
        proportion_3_4LG = count_3_4LG / total_dirs
    
    # Print results
    print(f"Proportion of '1LG' directories: {proportion_1LG:.4f}")
    print(f"Proportion of '2LG' directories: {proportion_2LG:.4f}")
    print(f"Proportion of '3+4LG' directories: {proportion_3_4LG:.4f}")

def count_children(dir_path):
    unique_names = set()
    
    # List directories in the specified directory
    for d in os.listdir(dir_path):
        full_path = os.path.join(dir_path, d)
        if os.path.isdir(full_path):
            # Extract the name before the first "_"
            unique_name = d.split('_')[0]
            unique_names.add(unique_name)
    
    # Count of unique names
    num_unique_children = len(unique_names)
    print(f"Number of unique children: {num_unique_children}")

count_wordlists(dir_path)
count_children(dir_path)
