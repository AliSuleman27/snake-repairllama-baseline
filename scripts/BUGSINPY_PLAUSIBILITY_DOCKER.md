# BugsInPy Plausibility — Docker pipeline (local Windows)

End-to-end guide for running plausibility tests on the trained-adapter
generations using the local `BugsInPy/` Docker setup. Replaces the
broken Colab+pyenv flow that produced false positives on `ansible/15`.

## Prerequisites

- Docker Desktop installed, running, WSL 2 backend enabled.
- ~30 GB free on `D:\` (Docker images + per-project conda envs +
  per-bug checkouts).
- ~6 GB RAM allocated to Docker (Settings → Resources → Memory).

## One-time setup

### 1. Build the image (~5 min first time, ~5 sec on rebuilds)

```powershell
cd D:\snake-repairllama-baseline\BugsInPy
docker compose build setup
```

The Dockerfile (already patched in commit `21deab7`) installs
`miniconda3` + the BugsInPy framework, runs `dos2unix` on framework
scripts (so Windows CRLF line endings don't break the shebangs), and
chmods the bin scripts executable.

Verify:
```powershell
docker images | findstr bugsinpy
# bugsinpy-setup    latest    ...    ~1 GB
```

### 2. Sanity check (~2 min) — confirms the framework works

```powershell
docker compose run --rm setup bugsinpy-testall -p thefuck:1:1
```

Expected output:
```
thefuck,1,buggy,fail
thefuck,1,fixed,pass
```

If you see `pass` for the fixed version, the pipeline is healthy. If
both come back `error`, see the troubleshooting section below.

## The pre-filter (already done — committed in `21deab7`)

The pre-filter cut 1610 generations on 161 bugs down to **460 patches
on 53 bugs** that actually need testing:

- 1059 patches skipped (failed our `splice-and-parse` compile metric)
- 53 patches skipped (within-bug duplicates)
- 98 of 151 reproducible bugs need NO checkout/compile
- 10 of 161 bugs excluded entirely (upstream marks them as
  non-reproducible — see `data/bugsinpy_eval_verified_reproducible.dropped.txt`)

Outputs are committed at:
- `data/bugsinpy_eval_verified_reproducible.jsonl`
- `results/bugsinpy_snakellama_run3_prefilter_reproducible/filtered_inference.jsonl`
- `results/bugsinpy_snakellama_run3_prefilter_reproducible/skipped_results.jsonl`
- `results/bugsinpy_snakellama_run3_prefilter_reproducible/dedup_map.json`

If you ever want to re-generate them (e.g. after a new inference run):
```powershell
python scripts\prefilter_for_plausibility.py filter `
    --eval data\bugsinpy_eval_verified_reproducible.jsonl `
    --inference results\bugsinpy_snakellama_run3.jsonl `
    --out-dir results\bugsinpy_snakellama_run3_prefilter_reproducible
```

## Running the plausibility tests

### Option A — Smoke test on one project first (recommended)

Run on `thefuck` only to confirm the runner works end-to-end.
~12 bugs × ~9 patches = ~110 tests, ~30 min:

```powershell
cd D:\snake-repairllama-baseline\BugsInPy
docker run --rm `
    -v "D:\snake-repairllama-baseline:/work" `
    bugsinpy-setup:latest `
    python /work/scripts/bugsinpy_plausibility_docker.py `
        --eval /work/data/bugsinpy_eval_verified_reproducible.jsonl `
        --inference /work/results/bugsinpy_snakellama_run3_prefilter_reproducible/filtered_inference.jsonl `
        --output /work/results/bugsinpy_snakellama_run3_plausibility_tested.jsonl `
        --projects thefuck
```

Watch the output. Each bug prints:
```
[thefuck/1] checkout ...
[thefuck/1] env + compile ...
[thefuck/1] testing 9 unique compiling patches ...
  [PASS] gen[0] -> pass
  [FAIL] gen[1] -> fail
  [PASS] gen[2] -> pass
  ...
```

If you see `[ERR ]` for everything, something's wrong with the
environment — paste the output and the corresponding
`/tmp/bugsinpy_logs/<project>_<bug>.log` (visible inside the
container; shut the container down with `docker ps` + `docker stop`
and inspect via `docker exec`).

### Option B — Full run (53 bugs, ~5 hours)

Once smoke-test looks healthy:

```powershell
cd D:\snake-repairllama-baseline\BugsInPy
docker run --rm `
    -v "D:\snake-repairllama-baseline:/work" `
    bugsinpy-setup:latest `
    python /work/scripts/bugsinpy_plausibility_docker.py `
        --eval /work/data/bugsinpy_eval_verified_reproducible.jsonl `
        --inference /work/results/bugsinpy_snakellama_run3_prefilter_reproducible/filtered_inference.jsonl `
        --output /work/results/bugsinpy_snakellama_run3_plausibility_tested.jsonl
```

The runner is **resumable**: re-running the same command picks up
where it left off. If your machine reboots or you Ctrl+C, just rerun.

### Phase budget per project (rough)

| Project | # bugs to test | Notes |
|---------|---------------|-------|
| thefuck | ~12 | Fast (small deps). ~5 min/bug. |
| youtube-dl | ~9 | Fast. ~5 min/bug. |
| pandas | ~6-8 | Slow (huge deps). ~20-30 min/bug. **Biggest budget.** |
| keras | ~4-5 | Slow (TF). ~15-20 min/bug. |
| black, fastapi, scrapy | ~3-5 each | Medium. ~10 min/bug. |
| cookiecutter, ansible, matplotlib, tqdm | ~1-2 each | Fast-medium. |

The pandas + keras conda compile is the dominant cost. Conda envs are
cached within a single `docker run`, so all bugs in pandas share the
same env after the first creation (~5-10 min).

## After the run finishes

### Merge tested + skipped + duplicates → final plausibility JSONL

```powershell
cd D:\snake-repairllama-baseline
python scripts\prefilter_for_plausibility.py merge `
    --filtered-inference results\bugsinpy_snakellama_run3_prefilter_reproducible\filtered_inference.jsonl `
    --tested results\bugsinpy_snakellama_run3_plausibility_tested.jsonl `
    --skipped results\bugsinpy_snakellama_run3_prefilter_reproducible\skipped_results.jsonl `
    --dedup-map results\bugsinpy_snakellama_run3_prefilter_reproducible\dedup_map.json `
    --out results\bugsinpy_snakellama_run3_plausibility.jsonl
```

This produces a JSONL with **one row per (bug, gen_idx)** for the
1510 generations on the 151 reproducible bugs (10 bugs were dropped
because the upstream BugsInPy can't reproduce them at all — they
contribute nothing to the denominator).

### Score with plausibility

```python
from src.metrics import evaluate_file, print_report
result = evaluate_file(
    inference_jsonl="results/bugsinpy_snakellama_run3.jsonl",
    eval_jsonl="data/bugsinpy_eval_verified_reproducible.jsonl",
    plausibility_jsonl="results/bugsinpy_snakellama_run3_plausibility.jsonl",
)
print_report("BugsInPy — Trained run3 (plausibility, reproducible-only)", result)
```

You'll get **plausible@1, plausible@3, plausible@10** added to the
existing report — the headline numbers for your thesis.

## Result file schema

Each row in `bugsinpy_snakellama_run3_plausibility_tested.jsonl`:

```json
{
  "bug_id": "thefuck/3",
  "gen_idx": 4,
  "compile_pass": true,
  "test_pass": false,
  "test_status": "fail",
  "stderr": "...last 1000 chars of test output..."
}
```

Status values:
- `pass` — test suite passed against our patch (plausible)
- `fail` — test suite ran and reported FAILED (model produced wrong fix)
- `error` — test infra problem (ImportError, command not found, etc.)
- `timeout` — test exceeded `--timeout-test` (default 180 s)
- `compile_fail_skipped` — pre-filtered, never tested (saved compute)
- `compile_failed` — bugsinpy-compile didn't produce a working venv
- `checkout_failed` — bugsinpy-checkout failed
- `<status>_via_duplicate` — result copied from the canonical sibling

## Troubleshooting

### Sanity check `bugsinpy-testall -p thefuck:1:1` returns `error,error`

Likely causes:
1. **Build cached old layers** — run `docker compose build --no-cache setup`.
2. **Docker memory too low** — bump to 6 GB in Docker Desktop → Resources.
3. **Disk full** — Docker writes to `C:\ProgramData\Docker\` or your
   configured path; needs ~25 GB free.

### A specific bug always errors but others pass

Check `/tmp/bugsinpy_logs/<project>_<bug>.log` inside the container.
Common reasons:
- Project requires native deps not in the image (rare).
- Bug's commit pulls in incompatible Python version with new conda
  defaults (occasionally pandas/scipy bugs).

If a bug is genuinely unreproducible, add it to
`data/bugsinpy_eval_verified_reproducible.dropped.txt` and re-run the
pre-filter to exclude it.

### Resume from where it left off

Just re-run the same `docker run` command. The script reads existing
rows in `--output` and skips `(bug_id, gen_idx)` pairs already
present. No flag needed.

### Shut down a stuck run

In another PowerShell window:
```powershell
docker ps | findstr bugsinpy
docker stop <container_id>
```

The output JSONL is appended in real time, so progress is preserved.

## What next

After plausibility is computed and merged, the final headline numbers
to add to `context.md` would look like:

```
BugsInPy — Trained run3 (plausibility, 151 reproducible bugs)
  Top-1  Plausible: ?? / 151 (??.?%)
  Top-3  Plausible: ?? / 151 (??.?%)
  Top-10 Plausible: ?? / 151 (??.?%)
```

These are the metrics RepairLLaMA reports as their main result —
strict test-execution-pass plausibility. Single-digit % is the
expected ballpark (RepairLLaMA's own numbers are similar).
