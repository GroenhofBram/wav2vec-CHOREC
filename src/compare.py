import os

def get_directories(path):
    """Get the list of directories in the given path."""
    return set([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])

def compare_directories(path1, path2):
    """Compare directories between two paths and return the differences."""
    dirs1 = get_directories(path1)
    dirs2 = get_directories(path2)

    only_in_path1 = dirs1 - dirs2
    only_in_path2 = dirs2 - dirs1

    return only_in_path1, only_in_path2

path1 = "D:\\repos\\wav2vec-CHOREC\\output\\GroNLP-TRAINING_SET-wav2vec2-dutch-large-ft-cgn"
path2 = "D:\\repos\\whisper-CHOREC\\output\\GroNLP-TRAINING-WHISPERwav2vec2-dutch-large-ft-cgn"

only_in_path1, only_in_path2 = compare_directories(path1, path2)

print("Directories only in path1:")
for dir in only_in_path1:
    print(dir)

print("\nDirectories only in path2:")
for dir in only_in_path2:
    print(dir)
