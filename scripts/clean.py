import sys, os, re, shutil

def delete_all_models(directory):
    for filename in os.listdir(directory):
        if filename != '.gitignore':
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def clean_checkpoints(directory):
    pattern = re.compile(r'_checkpoint_\d+')
    for root, _, files in os.walk(directory):
        for filename in files:
            if pattern.search(filename):
                file_path = os.path.join(root, filename)
                try:
                    os.unlink(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

if __name__ == "__main__":
    from config import MODEL_DIR
    if len(sys.argv) > 1 and sys.argv[1] == 'all':
        delete_all_models(MODEL_DIR)
    else:
        clean_checkpoints(MODEL_DIR)

# TODO: For Safty, rename last checkpoint to {model_name}.pt before deleting all checkpoints
