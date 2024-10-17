import os
import shutil
import stat

def main():
    remove_dir = r"D:\repos\wav2vec-CHOREC\output\GroNLP-TRAINING_SET-wav2vec2-dutch-large-ft-cgn"
    if os.path.exists(remove_dir):
        remove_empty_dirs(remove_dir)
    else:
        print(f"The directory {remove_dir} does not exist.")


def handle_remove_readonly(func, path, exc_info):
    # Change the permission to writable and then retry the operation
    os.chmod(path, stat.S_IWRITE)
    func(path)

def remove_empty_dirs(path):
    # Walk the directory from the bottom up
    for root, dirs, files in os.walk(path, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            # If the directory is empty, remove it
            os.rmdir(dir_path)
            print(f"Removed empty directory: {dir_path}")

main()