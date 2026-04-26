#!/usr/bin/env bash
#
# setup_bugsinpy.sh
# -----------------
# One-time setup for BugsInPy plausibility testing on Linux (Colab/WSL/Ubuntu).
#
# What it does:
#   1. Clones BugsInPy framework
#   2. Installs pyenv (manages multiple Python versions)
#   3. Installs Python build deps (libssl, libffi, etc.) via apt
#   4. Installs Python versions commonly needed by BugsInPy (3.6, 3.7, 3.8)
#   5. Patches a known shell quirk in bugsinpy-test (`source` vs `.`)
#
# Usage (Colab):
#     !bash scripts/setup_bugsinpy.sh /content/BugsInPy /content/.pyenv
#
# After this runs once, set PATH and PYENV_ROOT in subsequent cells:
#     export BUGSINPY_DIR=/content/BugsInPy
#     export PYENV_ROOT=/content/.pyenv
#     export PATH="$BUGSINPY_DIR/framework/bin:$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

set -e

BUGSINPY_DIR=${1:-/content/BugsInPy}
PYENV_ROOT=${2:-$HOME/.pyenv}
PYTHON_VERSIONS=${PYTHON_VERSIONS:-"3.6.15 3.7.17 3.8.18"}

echo "===================================================="
echo "  BugsInPy plausibility setup"
echo "  BUGSINPY_DIR    = $BUGSINPY_DIR"
echo "  PYENV_ROOT      = $PYENV_ROOT"
echo "  PYTHON_VERSIONS = $PYTHON_VERSIONS"
echo "===================================================="

# --- 1. apt deps (idempotent) -------------------------------------------------
if [ -f /etc/debian_version ]; then
    echo "[1/4] Installing apt build deps ..."
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        make build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev wget curl llvm \
        libncurses5-dev libncursesw5-dev xz-utils tk-dev \
        libffi-dev liblzma-dev git \
        > /dev/null
fi

# --- 2. clone BugsInPy --------------------------------------------------------
if [ ! -d "$BUGSINPY_DIR/.git" ]; then
    echo "[2/4] Cloning BugsInPy ..."
    git clone --depth 1 https://github.com/soarsmu/BugsInPy.git "$BUGSINPY_DIR"
else
    echo "[2/4] BugsInPy already cloned at $BUGSINPY_DIR"
fi
chmod +x "$BUGSINPY_DIR/framework/bin/"* || true

# --- 3. install pyenv ---------------------------------------------------------
if [ ! -d "$PYENV_ROOT" ]; then
    echo "[3/4] Installing pyenv ..."
    git clone --depth 1 https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
else
    echo "[3/4] pyenv already at $PYENV_ROOT"
fi
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# --- 4. install Python versions ----------------------------------------------
echo "[4/4] Installing Python versions: $PYTHON_VERSIONS"
for v in $PYTHON_VERSIONS; do
    if [ -d "$PYENV_ROOT/versions/$v" ]; then
        echo "    - $v already installed"
    else
        echo "    - Installing Python $v (5-10 min) ..."
        pyenv install -s "$v" \
          || echo "    [WARN] pyenv install $v failed; continuing"
    fi
done

echo ""
echo "===================================================="
echo "  DONE"
echo "  In subsequent cells, set:"
echo "    export BUGSINPY_DIR=$BUGSINPY_DIR"
echo "    export PYENV_ROOT=$PYENV_ROOT"
echo "===================================================="
