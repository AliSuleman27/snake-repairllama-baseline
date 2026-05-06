"""
upload_checkpoint.py
--------------------
Push the local LoRA checkpoint to a private HuggingFace Hub repo so we can
re-download it onto any RunPod instance.

Run from Windows:    python upload_checkpoint.py
"""
import os
from pathlib import Path

# === EDIT THESE ============================================================
HF_TOKEN  = ""
HF_REPO   = "alisuleman525/snake-repairllama-checkpoint-400"   # change user if needed
LOCAL_DIR = r"C:\Users\3TEE\Downloads\train1\train\output\snake-repairllama-7b-fim-r16\checkpoint-400"
# ============================================================================

if not Path(LOCAL_DIR).is_dir():
    raise SystemExit(f"Not a directory: {LOCAL_DIR}")

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("Installing huggingface_hub ...")
    os.system("pip install -q huggingface_hub")
    from huggingface_hub import HfApi, login

login(token=HF_TOKEN, add_to_git_credential=False)
api = HfApi()

# Create the private repo (idempotent — won't fail if it already exists)
api.create_repo(repo_id=HF_REPO, repo_type="model", private=True, exist_ok=True)
print(f"Repo ready: https://huggingface.co/{HF_REPO}")

# Upload the folder
print(f"Uploading {LOCAL_DIR} ...")
api.upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=HF_REPO,
    repo_type="model",
    commit_message="Upload checkpoint-400 from local",
)
print("Done.")
