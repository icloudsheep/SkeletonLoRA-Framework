# SecLoRA End-to-End Module

This directory is owned by the `SecLoRA_EndToEnd` branch. It adapts the
PC-DMCFE selective two-server implementation to the framework without changing
the shared `client`, `server`, `runtime`, or `utils` packages.

## Boundary

```text
PEFT state_dict
  -> canonical A/B layer manifest
  -> persistent native SEL-2S session
  -> integer C/M/S skeletons
  -> fixed-point normalization
  -> low-rank rank-r factorization
  -> PEFT-compatible state_dict
```

The native session is initialized once per training process. Global setup and
the reusable BSGS table must not be rebuilt for every layer. Layers are streamed
through the native session so encryption precomputation can be released after
each layer.

The existing compact representation for quantized all-zero A columns and B rows
is intentionally preserved.

## SEL-2S Data Flow

- Quantization is `round(2^sfp * clip(x, -xmax, xmax))`.
- The public BSGS bound is `M = ceil(2^sfp * xmax)^2 * K * R`.
- `S_P` receives only clear B rows and clear A columns as signed int64 factors.
- `S_D` receives only PC-MCFE objects for the protected prefix and the public
  pivot candidate pool.
- The candidate pool contains at most `2*K*R` deterministic, spread-out rows
  and columns from the clear region. Gaussian pivoting over this public block
  produces nested nonsingular row/column choices; it is not used to estimate
  the rank of the complete plaintext block.
- Skeleton search starts at the configured LoRA rank `R` and increases one rank
  at a time, up to `K*R`. Previously decrypted entries are cached. Increasing
  the rank by one therefore adds only one C column and one S row, including
  `encrypted_B_rows + encrypted_A_cols` new bounded recoveries at `S_D`.
- Each client uploads one additional compressed PC-DMCFE pair for the public
  random projection `beta^T * B_i * A_i * alpha`. This is needed because SEL-2S
  intentionally omits ciphertext labels for the clear factor slices.
- For every candidate rank, the output holder computes
  `beta^T * C * M^-1 * S * alpha` in the scalar field and compares its group
  encoding directly with the true encrypted projection. Projection checking
  does not run BSGS. With uniform field challenges, an incorrect reconstruction
  passes one check with probability at most `2/p`. The first passing rank is
  returned.
- `S_D` decrypts only the protected cells required by C and S. M and the clear
  portions of C/S are computed by `S_P`. If no candidate rank passes, the round
  fails explicitly instead of returning a potentially incomplete aggregate.

The two servers are enforced as separate payloads and computation paths inside
one process. Physical process separation and transport serialization remain
future deployment work. Public pivot candidates must still contain a
nonsingular intersection large enough for the successful skeleton rank.

## Native Contract

The compiled module must expose:

```python
SelectiveTwoServerSession(
    num_clients: int,
    rank: int,
    ratio: float,
    sfp: int,
    xmax: float,
    threads: int,
)

session.encrypt_client(client_id, round_id, layers) -> NativeClientUpdate
session.aggregate_round(round_id, updates) -> list[NativeLayerSkeleton]
```

`NativeClientUpdate` exposes `serialized_size_bytes`. Each returned layer
exposes `layer_id`, `c`, `m`, and `s`, where C/M/S contain signed decoded
fixed-point integers. It also exposes:

- `selected_rank`: first rank in `R..K*R` whose projection check passed.
- `projection_checks`: number of attempted ranks.
- `decrypted_cells`: unique protected C/S cells recovered with BSGS.

The Python layer divides the reconstructed sum by
`num_clients * 2^(2*sfp)` and compresses it back to the configured LoRA rank
without materializing the full product matrix.

## Build And Run

Reuse the framework's `skeleton_lora_fe` environment. On a fresh AutoDL
machine, install any missing system build tools, then run the setup script:

```bash
conda activate skeleton_lora_fe

apt-get update
apt-get install -y build-essential cmake git libgmp-dev

bash seclora/setup_autodl.sh
bash seclora/verify_autodl.sh
```

If the system packages are already installed, the `apt-get` commands can be
skipped. Without root access, use:

```bash
conda install -y -c conda-forge cmake ninja cxx-compiler gmp
bash seclora/setup_autodl.sh
```

For later source changes, rebuilding does not require reinstalling the
environment:

```bash
bash seclora/native/build.sh clean
bash seclora/verify_autodl.sh
```

After the smoke test passes, start the real OpenLLaMA/Dolly run:

```bash
python main.py --config configs/seclora_end_to_end.yaml
```

### AutoDL libstdc++ mismatch

If importing `optree`, Torch, Transformers, or PEFT reports that
`GLIBCXX_3.4.31` is missing, do not replace or symlink the system library.
Install the compatible runtime inside the existing Conda environment:

```bash
conda activate skeleton_lora_fe
conda install -y -c conda-forge "libstdcxx-ng>=13" "libgcc-ng>=13"
conda env config vars set \
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
conda deactivate
conda activate skeleton_lora_fe
```

Verify the repaired framework environment before rebuilding:

```bash
python -c "import optree, torch, transformers, peft; print('imports: OK')"
```

The build script pins herumi/mcl to commit `7af8ea7`. Set `PYTHON_BIN` when the
active interpreter is not named `python`, or `SECLORA_MCL_DIR` to reuse an
existing checkout at that exact revision.
