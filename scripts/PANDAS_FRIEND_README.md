# Running pandas plausibility on someone else's PC

Hand this to your friend with the repo. They run **one command** and the
script handles everything else.

## Prereqs (on their machine)

- **Docker Desktop** installed and running (Windows: WSL2 backend; macOS/Linux: native)
- **Python 3.10+** in PATH
- **Git** in PATH
- **At least 50 GB free** on whatever drive Docker uses (pandas builds 30+
  conda envs)
- **Stable internet** — first run downloads `continuumio/miniconda3:23.3.1-0`
  (~500 MB) and pip-installs ~100 packages per pandas bug

## One-shot run (Windows PowerShell)

```powershell
git clone https://github.com/AliSuleman27/snake-repairllama-baseline.git
cd snake-repairllama-baseline
git submodule update --init --recursive
pwsh -ExecutionPolicy Bypass -File .\scripts\run_pandas_on_friend_pc.ps1
```

That's it. The script:

1. Verifies Docker is running
2. Initializes the BugsInPy submodule if missing
3. Builds the `bugsinpy-setup` Docker image (cached on subsequent runs)
4. Creates `bugsinpy_envs` and `bugsinpy_work` named volumes
5. Splits the 45 pandas bugs round-robin across 4 worker containers
6. Runs them in parallel, prints final pass@10 + per-bug breakdown

## ETA

- First run: **~6 hours wall time** on a typical workstation
- Pandas bugs are heavy: each needs to compile NumPy/pandas C extensions
- Workers run in parallel (4 dockers concurrent)
- Subsequent runs (e.g. against a different model's generations) reuse the
  cached envs and finish in **~45 min**

## What to send back

After the run completes, send back:

```
results/all_docker_runs_result/bugsinpy_snakellama_pandas_gold.jsonl
results/all_docker_runs_result/bugsinpy_snakellama_pandas_gen_part{1..4}.jsonl
results/all_docker_runs_result/bugsinpy_snakellama_pandas_gold_part{1..4}.jsonl
results/snakellama/bugsinpy_snakellama_pandas_plausibility.jsonl
```

Or just zip these two directories — they also include per-worker logs which
are useful for debugging:

```
results/all_docker_runs_result/bugsinpy_snakellama_pandas_*
results/snakellama/bugsinpy_snakellama_pandas_*
```

## If something goes wrong

- **Docker daemon not running**: the script aborts immediately. Open Docker
  Desktop and re-run.
- **Out of disk space mid-run**: workers will fail with `compile_failed`.
  Free up space, then run the script again — it's resumable (skips bugs
  already in the part files).
- **Worker exits with rc=137**: usually OOM kill. Reduce concurrency by
  passing `--n-workers 2` to `scripts/run_pandas_4workers.py` (or edit the
  ps1 wrapper to call it that way).
- **Network drop during conda install**: re-run; resume picks up.

## What to NOT do

- Don't delete the named volumes (`bugsinpy_envs`, `bugsinpy_work`) between
  runs. They cache 20+ GB of conda env state and project checkouts.
- Don't run on C: drive if it has < 60 GB free. Use Docker Desktop's
  Settings → Resources → Advanced → "Disk image location" to put it on a
  larger drive first.
