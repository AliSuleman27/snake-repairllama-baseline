# run_pandas_on_friend_pc.ps1
# =================================================================
# One-shot setup + run for the BugsInPy pandas plausibility split.
# Designed for a clean machine (Muneeb's): assumes only Docker
# Desktop and Python 3.10+ are installed.
#
# What it does (in order):
#   1. Sanity-check Docker Desktop is running
#   2. Make sure the repo + BugsInPy submodule are present
#   3. Build the bugsinpy-setup image (idempotent)
#   4. Create the named volumes (idempotent)
#   5. Launch the 4-way pandas orchestrator
#   6. Print final pass@10 + per-bug breakdown
#
# Run from an elevated PowerShell at the repo root:
#     pwsh -ExecutionPolicy Bypass -File .\scripts\run_pandas_on_friend_pc.ps1
# =================================================================

$ErrorActionPreference = "Stop"

# ---- 0. Locate repo root (this script lives in scripts/) ----
$RepoRoot = Resolve-Path "$PSScriptRoot\.."
Set-Location $RepoRoot
Write-Host "Repo root: $RepoRoot" -ForegroundColor Cyan

# ---- 1. Docker check ----
Write-Host "`n[1/5] Checking Docker..." -ForegroundColor Yellow
try {
    docker version --format '{{.Server.Version}}' | Out-Null
    Write-Host "  Docker daemon OK" -ForegroundColor Green
} catch {
    Write-Host "  Docker daemon NOT running. Open Docker Desktop and try again." -ForegroundColor Red
    exit 1
}

# ---- 2. BugsInPy submodule ----
Write-Host "`n[2/5] Checking BugsInPy submodule..." -ForegroundColor Yellow
if (-not (Test-Path "BugsInPy\Dockerfile")) {
    Write-Host "  BugsInPy missing. Initializing submodule..."
    git submodule update --init --recursive
}
if (-not (Test-Path "BugsInPy\Dockerfile")) {
    Write-Host "  BugsInPy still missing after submodule init. Aborting." -ForegroundColor Red
    exit 1
}
Write-Host "  BugsInPy present" -ForegroundColor Green

# ---- 3. Build image (idempotent — uses Docker layer cache) ----
Write-Host "`n[3/5] Building bugsinpy-setup image..." -ForegroundColor Yellow
Push-Location BugsInPy
docker build -q -t bugsinpy-setup:latest . | Out-Host
Pop-Location
Write-Host "  Image ready" -ForegroundColor Green

# ---- 4. Named volumes (persist across runs) ----
Write-Host "`n[4/5] Creating named volumes..." -ForegroundColor Yellow
docker volume create bugsinpy_envs | Out-Null
docker volume create bugsinpy_work | Out-Null
Write-Host "  bugsinpy_envs + bugsinpy_work ready" -ForegroundColor Green

# ---- 5. Required input files ----
Write-Host "`n[pre-run] Verifying required input files..." -ForegroundColor Yellow
$Inference = "results/snakellama_model_generations/bugsinpy_snakellama_run3.jsonl"
$Eval = "data/bugsinpy_eval_verified.jsonl"
foreach ($f in @($Inference, $Eval)) {
    if (-not (Test-Path $f)) {
        Write-Host "  Required file MISSING: $f" -ForegroundColor Red
        Write-Host "  Pull latest from main, or copy this file from the run repo." -ForegroundColor Red
        exit 1
    }
}
Write-Host "  All inputs present" -ForegroundColor Green

# ---- 6. Launch pandas orchestrator ----
Write-Host "`n[5/5] Launching 4-way pandas plausibility run..." -ForegroundColor Yellow
Write-Host "  ETA: ~6h on a typical workstation (45 bugs split 4-way)." -ForegroundColor DarkYellow
Write-Host "  Per-worker logs in results\snakellama_full\pandas_worker{1..4}.log" -ForegroundColor DarkYellow
Write-Host ""

$env:MSYS_NO_PATHCONV = "1"
python scripts\run_pandas_4workers.py `
    --eval "/work/$Eval" `
    --inference "/work/$Inference" `
    --out-dir "results\snakellama_full"

# ---- 7. Final report ----
Write-Host "`n=== FINAL REPORT ===" -ForegroundColor Cyan
$report = @"
import json, glob, collections
runnable = set(); evald = set()
for f in sorted(glob.glob('results/snakellama_full/pandas_gold_part*.jsonl')):
    with open(f, encoding='utf-8') as fh:
        for l in fh:
            if l.strip():
                r = json.loads(l)
                evald.add(r['bug_id'])
                if r['gold_test_status'] == 'pass':
                    runnable.add(r['bug_id'])
gp = collections.Counter()
for f in sorted(glob.glob('results/snakellama_full/pandas_gen_part*.jsonl')):
    with open(f, encoding='utf-8') as fh:
        for l in fh:
            if l.strip():
                r = json.loads(l)
                if r['bug_id'] in runnable and r['test_status'].replace('_via_dedup','') == 'pass':
                    gp[r['bug_id']] += 1
plaus = sum(1 for b in runnable if gp[b] > 0)
print(f'Pandas evaluated:    {len(evald)}/45')
print(f'Reproducible (gold): {len(runnable)}')
print(f'pass@10:             {plaus}/{len(runnable)} = {100*plaus/max(len(runnable),1):.1f}%')
print()
print('Bugs with at least 1 model pass:')
for b, n in sorted(gp.items(), key=lambda x: (-x[1], x[0])):
    print(f'  {b:25s} {n}/10')
"@
python -c $report
Write-Host "`nDone. Send back the results\snakellama_full\pandas_*.jsonl files." -ForegroundColor Green
