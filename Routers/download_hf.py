from huggingface_hub import snapshot_download
import os

# Download with progress bar
dataset_path = snapshot_download(
    repo_id='nics-efc/R2R_router_collections',
    repo_type='model',
    local_dir='./R2R_router',
    local_dir_use_symlinks=False,
    # allow_patterns=["**/dev*", "**/validation*"],  # Download files starting with dev OR val
    tqdm_class=None,
    resume_download=True
)
print(f'Dataset downloaded to: {dataset_path}')
