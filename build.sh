#!/usr/bin/env bash
# Build the solver. Uses nvcc when available, otherwise a plain C++ CPU build.
#
#   ./build.sh            normal build
#   ./build.sh fermat     use Fermat inversion instead of the default
#   ./build.sh eea        use extended-Euclid inversion
set -e
cd "$(dirname "$0")"
SRC=src/rho.cu

EXTRA=""
case "${1:-}" in
    fermat) EXTRA="-DINV_MODE=2" ;;
    eea)    EXTRA="-DINV_MODE=1" ;;
esac

if command -v nvcc >/dev/null 2>&1; then
    echo "nvcc found -> CUDA build (rho_toy, rho)"
    nvcc -O3 -std=c++17 -arch=native $EXTRA -DUSE_TOY "$SRC" -o rho_toy
    nvcc -O3 -std=c++17 -arch=native $EXTRA "$SRC" -o rho
    echo
    echo "build ok. try:"
    echo "  ./rho_toy selftest && ./rho_toy cpu 12"
    echo "  ./rho gpu 22"
else
    echo "nvcc not found -> CPU fallback (rho_cpu, rho_toy)"
    CXX="${CXX:-clang++}"
    $CXX -O2 -x c++ "$SRC" -o rho_cpu
    $CXX -O2 -x c++ -DUSE_TOY "$SRC" -o rho_toy
    echo
    echo "build ok (CPU). try:"
    echo "  ./rho_toy selftest && ./rho_toy cpu 12"
    echo "  ./rho_cpu selftest"
fi
