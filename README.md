# ecdlp-rho

This program (as you might have guesses or not) was used for stuff that is not allowed to be published on github.
Therefore take it as not a program or a tool that is usable but a simple example of how a certain size scalar key
can be found using pollard rho solution as mentioned below.

A parallel **Pollard-rho** solver for the elliptic-curve discrete logarithm problem
(ECDLP), in one source that builds as CUDA or plain C++.

Given two points `C1`, `C2` in a prime-order group, it finds the scalar `k` with

```
C2 = [k] C1            ( mod ORD )
```

The bundled instance works in a prime-order subgroup of the curve over `F_p[i]`
(`i^2 = -1`):

```
E'(F_p^2) : y^2 = x^3 + (b0 + b1 i)
```

The curve, generator, and target are produced by `gen.py` (a CM-method search for a
random prime-order instance). Two instances are generated:

| instance | field | group order | use |
|----------|-------|-------------|-----|
| *toy*    | ~20-bit | ~39-bit | fast self-test (CPU, milliseconds) |
| *sample* | ~44-bit | ~87-bit | the real run (GPU) |

`params.h` is regenerated from `gen.py`; the toy instance ships with its known scalar
so the build can prove itself before any long run.

## Files

| file | purpose |
|------|---------|
| `src/gen.py` | generates `src/params.h` (curve, generator, target, toy key, test vectors) |
| `src/params.h` | auto-generated constants and F_p / F_p^2 / point test vectors |
| `src/rho.cu` | the solver: CUDA, C++ CPU-fallback, and pooling, one source |
| `build.sh` / `build.bat` | build scripts (CUDA if available, else CPU) |
| `run_pool.sh` | launches N GPU instances sharing one distinguished-point pool |

## Build

CUDA (needs the CUDA Toolkit; current devices may need `-arch=native` or a matching
`sm_XX`):

```
./build.sh                 # -> rho_toy, rho
```

CPU only (any host, for validation):

```
clang++ -O2 -x c++ src/rho.cu -o rho_cpu
clang++ -O2 -x c++ -DUSE_TOY src/rho.cu -o rho_toy
```

On Windows use `build.bat` (uses `nvcc` if present, else MSVC / or a CPU fallback).

## Run

1. Verify the field and point arithmetic (must print `ok`):

```
./rho_toy selftest
./rho     selftest
```

2. Verify the whole solver on the toy instance — it must recover the known scalar
   printed by `gen.py` (runs in well under a second on CPU):

```
./rho_toy cpu 12
```

3. The sample run. `D` is the distinguished-point size in bits: larger `D` stores
   fewer points but makes walks longer. `D` around 20–22 is a good range.

```
./rho gpu 22
```

   On success it prints `sk = 0x...` and writes `sk.txt`.

Other modes:

```
./rho speed [blocks threads budget]   # throughput test (steps/s)
./rho diag  [D blocks threads budget] # per-round diagnostics
./rho pooltest                        # plant a cross-file collision (CPU, must PASS)
./rho poolscan                        # merge the pool and scan for a collision
```

`gpu` records distinguished points on the device and resolves collisions on the host;
`cpu` is single-threaded and intended for validation and the toy instance.

## Method

* **Walk.** `X <- X + R_j` where `j = hash(X.x) mod W` and `R_j = u_j*C1 + v_j*C2`
  is a fixed precomputed table. Each walk carries `(a, b)` so that `X = a*C1 + b*C2`
  is invariant; the table is seeded once so every instance walks the *same* `f`.
* **Distinguished points.** A walk stops when `X.x` is `D`-bit distinguished, and the
  point plus its `(a, b)` is stored. Distinguished points are restricted to a canonical
  half by `y` so an `x`-collision cannot be confused with `X` vs `-X`.
* **Collision.** Two walks reaching the same distinguished point give
  `X = a1*C1 + b1*C2 = a2*C1 + b2*C2`, hence
  `k = (a1 - a2) * (b2 - b1)^-1 mod ORD`, verified by checking `[k]C1 == C2`.
* **Walks** have a step budget so they can escape distinguished-point-free cycles; the
  store persists across walks so different walks still merge.

### Multiple instances / GPUs

Independent instances give only a `sqrt(N)` birthday speedup. For a **linear** speedup
they must share the distinguished-point store so a collision between two instances is
actually detected. Set a distinct `RHO_SEED` per instance and point them all at a
shared `RHO_POOL_DIR`; each instance appends its points to `pool_<seed>.bin` and merges
them once per round.

```
./run_pool.sh 8 ./rho 22
```

If the instances do not share a filesystem, rsync the pool directories between hosts
every few minutes.

### Environment variables

```
RHO_SEED        per-instance id (also diversifies start points)
RHO_POOL=1      enable the shared distinguished-point pool
RHO_POOL_MAX=N  number of pool_<i>.bin files to merge
RHO_POOL_DIR    directory holding the pool files
RHO_BUDGET      real point-adds per thread per round
RHO_STEP        per-walk step cap
```

## Notes

* The instance in `params.h` is deliberately small (a research/test-sized group). The
  solver scales to larger groups by regenerating `params.h` with a bigger
  `SAMPLE_BITS`, at the cost of `~sqrt(order)` work.
* `gen.py` uses the CM method for `j = 0` curves to find a `p` and twist with a
  prime-order subgroup, then emits the four `F_p[i]` coordinates of `C1` and `C2`.

## License

MIT — see `LICENSE`.
