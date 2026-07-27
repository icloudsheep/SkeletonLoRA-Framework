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
  and columns from the clear region. This keeps the end-to-end upload one-pass.
- `S_P` verifies that the candidate block captures the rank of the complete
  clear block. The round fails explicitly if it does not.
- `S_D` decrypts only the protected cells required by C and S. M and the clear
  portions of C/S are computed by `S_P`.

The two servers are enforced as separate payloads and computation paths inside
one process. Physical process separation and transport serialization remain
future deployment work. As in the standalone SEL-2S implementation, exact CUR
recovery assumes that the clear block captures the aggregate matrix rank.

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
fixed-point integers.

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

The build script pins herumi/mcl to commit `7af8ea7`. Set `PYTHON_BIN` when the
active interpreter is not named `python`, or `SECLORA_MCL_DIR` to reuse an
existing checkout at that exact revision.
