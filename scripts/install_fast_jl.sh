#!/usr/bin/env bash
# Build the fast_jl CUDA extension (used only by `catt attribute --job trak`) into
# the project venv.
#
# fast_jl compiles against the torch in the venv (no build isolation), so it needs
# a full CUDA toolkit (nvcc and the cuBLAS/cuSPARSE/cuSOLVER headers) of torch's
# CUDA major version (12 for torch 2.6.0+cu124). The script uses $CUDA_HOME if
# set, else the best of the nvcc on PATH, the active conda env and
# /usr/local/cuda*: CUDA 12.4 if there is one, else the nearest other 12.x
# (torch accepts a minor-version mismatch; the kernel test at the end checks the
# build runs). Without any, install CUDA 12.4 into a conda env (no root needed,
# about 6 GB):
#   conda create -n cuda124 -c nvidia/label/cuda-12.4.1 -c conda-forge cuda-toolkit
#   CUDA_HOME=$(conda run -n cuda124 printenv CONDA_PREFIX) bash scripts/install_fast_jl.sh
# Afterwards keep `--extra jl` on every `uv sync`, or uv removes it again.
set -euo pipefail
shopt -s nullglob
cd "$(dirname "$0")/.."
uv sync --extra build
want=$(.venv/bin/python -c "import torch; print(torch.version.cuda)")

release() { { "$1/bin/nvcc" --version 2>/dev/null || true; } | sed -n 's/.*release \([0-9.]*\),.*/\1/p'; }

if [[ -n "${CUDA_HOME:-}" ]]; then
  candidates=("$CUDA_HOME")
else
  candidates=()
  if command -v nvcc > /dev/null; then
    candidates+=("$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")")
  fi
  candidates+=(${CONDA_PREFIX:+"$CONDA_PREFIX"} "/usr/local/cuda-$want" /usr/local/cuda /usr/local/cuda-*)
fi
exact="" near="" near_v="" near_gap=999 seen="" checked=" "
for c in "${candidates[@]}"; do
  real=$(readlink -f "$c" 2>/dev/null || echo "$c")
  [[ "$checked" == *" $real "* ]] && continue
  checked+="$real "
  v=$(release "$c")
  [[ -z "$v" ]] && continue
  seen+="  $c (CUDA $v)"$'\n'
  if [[ "$v" == "$want" ]]; then exact=$c; break; fi
  if [[ "${v%%.*}" == "${want%%.*}" ]]; then
    m=${v#*.} w=${want#*.}
    gap=$(( m > w ? m - w : w - m ))
    if (( gap < near_gap )); then near=$c near_v=$v near_gap=$gap; fi
  fi
done
if [[ -n "$exact" ]]; then
  CUDA_HOME=$exact
  echo "building fast_jl with CUDA $want at $CUDA_HOME"
elif [[ -n "$near" ]]; then
  CUDA_HOME=$near
  echo "building fast_jl with CUDA $near_v at $CUDA_HOME (no CUDA $want found; torch was" \
       "built with $want and accepts another ${want%%.*}.x)"
else
  echo "no CUDA ${want%%.*}.x toolkit found (torch $(.venv/bin/python -c 'import torch; print(torch.__version__)') needs one, ideally $want)." >&2
  [[ -n "$seen" ]] && printf 'Found only:\n%s' "$seen" >&2
  cat >&2 <<EOF
Point CUDA_HOME at a CUDA ${want%%.*}.x install (e.g. after \`module load cuda/$want\`), or
install CUDA $want into a conda env (no root needed, about 6 GB):
  conda create -n cuda${want//./} -c nvidia/label/cuda-$want.1 -c conda-forge cuda-toolkit
  CUDA_HOME=\$(conda run -n cuda${want//./} printenv CONDA_PREFIX) bash scripts/install_fast_jl.sh
EOF
  exit 1
fi
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
# uv's wheel cache does not key on the CUDA toolkit or torch, so always rebuild
uv cache clean fast-jl
uv sync --extra jl --reinstall-package fast-jl
.venv/bin/python - <<'EOF'
import torch, fast_jl
print("fast_jl OK, torch", torch.__version__)
if torch.cuda.is_available():
    x = torch.randn(32, 4096, device="cuda").half()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    y = fast_jl.project_rademacher_32(x, 512, 0, sms)
    torch.cuda.synchronize()
    print(f"projected {tuple(x.shape)} -> {tuple(y.shape)} on {torch.cuda.get_device_name()}")
EOF
