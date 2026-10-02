#!/usr/bin/env bash
# Launch N GPU instances of the solver that share one distinguished-point pool.
# Whichever instance finds the scalar writes ./sk.txt; the rest are then stopped.
#
#   ./run_pool.sh [N] [binary] [D]
#     N       number of GPU instances (default: #GPUs from nvidia-smi, else 8)
#     binary  solver binary                    (default: ./rho)
#     D       distinguished-point bits         (default: 22)
#
# Overrides: RHO_POOL_DIR (default ./pool), RHO_POOL_EXTRA (extra solver args)
set -u

N="${1:-}"
BIN="${2:-./rho}"
D="${3:-22}"

if [ -z "$N" ] && command -v nvidia-smi >/dev/null 2>&1; then
    N="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
fi
if [ -z "$N" ] || [ "$N" = 0 ]; then N=8; fi

POOL_DIR="${RHO_POOL_DIR:-$(pwd)/pool}"
mkdir -p "$POOL_DIR"
export RHO_POOL=1 RHO_POOL_MAX="$N" RHO_POOL_DIR="$POOL_DIR"
rm -f sk.txt

echo "launching $N instance(s):  $BIN gpu $D"
echo "pool dir: $POOL_DIR"

pids=()
for i in $(seq 0 $((N - 1))); do
    RHO_SEED="$i" CUDA_VISIBLE_DEVICES="$i" \
        "$BIN" gpu "$D" ${RHO_POOL_EXTRA:-} > "log_seed_$i.txt" 2>&1 &
    pids+=("$!")
    echo "  seed $i -> log_seed_$i.txt (pid $!)"
done

# stop everyone the moment a scalar lands
while :; do
    if [ -s sk.txt ]; then
        echo; echo "=== sk.txt ==="; cat sk.txt; break
    fi
    alive=0
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    if [ "$alive" = 0 ]; then echo; echo "all instances exited"; break; fi
    sleep 10
done

for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done
wait 2>/dev/null

# Safety net: a genuine collision can sit in the pool files even if no instance
# scanned the full union before being stopped (per-round scans race).
if [ ! -s sk.txt ]; then
    echo; echo "=== final full-union poolscan ==="
    RHO_POOL=1 RHO_POOL_MAX="$N" RHO_POOL_DIR="$POOL_DIR" "$BIN" poolscan
fi
