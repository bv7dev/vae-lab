import os, re

def cleanup_models_directory(directory):
    for filename in os.listdir(directory):
        if filename != '.gitignore':
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def cleanup_checkpoints(directory):
    pattern = re.compile(r'_checkpoint_\d+')
    for filename in os.listdir(directory):
        if pattern.search(filename):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

if __name__ == "__main__":
    from config import MODEL_DIR
    cleanup_checkpoints(MODEL_DIR)
