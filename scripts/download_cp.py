"""
download_checkpoint.py
----------------------
Pull the previously uploaded checkpoint into the structure the trainer expects.
Run on RunPod:    python download_checkpoint.py
"""
import os
from pathlib import Path

# === EDIT THESE ============================================================
HF_TOKEN  = ""   # read access is enough
HF_REPO   = "alisuleman525/snake-repairllama-checkpoint-400"
LOCAL_DIR = "/workspace/snake-repairllama-baseline/train/output/snake-repairllama-7b-fim-r16/checkpoint-400"
# ============================================================================

try:
    from huggingface_hub import login, snapshot_download
except ImportError:
    os.system("pip install -q huggingface_hub")
    from huggingface_hub import login, snapshot_download

login(token=HF_TOKEN, add_to_git_credential=False)

Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)
print(f"Downloading {HF_REPO} -> {LOCAL_DIR}")
snapshot_download(
    repo_id=HF_REPO,
    repo_type="model",
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
)
print("\nFiles in checkpoint:")
for p in sorted(Path(LOCAL_DIR).iterdir()):
    print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
