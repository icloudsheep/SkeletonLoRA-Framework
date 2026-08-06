# SecLoRA End-to-End Module

This directory is owned by the `SecLoRA_EndToEnd` branch. It provides a shared
end-to-end implementation for `SEL-2S` and `FULL+SK` using the same training,
fixed-point, PC-DMCFE, skeleton, timing, and serialized-payload pipeline.

## Boundary

```text
PEFT state_dict
  -> canonical A/B layer manifest
  -> persistent native SecLoRA session (`SEL-2S` or `FULL+SK`)
  -> integer C/M/S skeletons
  -> fixed-point normalization
  -> low-rank rank-r factorization
  -> PEFT-compatible state_dict
```

The native session is initialized once per training process. Global setup and
the reusable BSGS table must not be rebuilt for every layer. Layers are streamed
through the native session so encryption precomputation can be released after
each layer.

The native core uses the paper-aligned two-coordinate mask construction: two
independent ABG19/ALS16 DMCFE instances recover the weighted A-side masks, and
the FH-IPFE vector has dimension `2R+3`. Quantized all-zero A columns and B rows
use the same standard ciphertext/key shape as nonzero vectors; no zero-structure
flag or compact zero encoding is transmitted.

## SEL-2S Data Flow

- Quantization is `round(2^sfp * clip(x, -xmax, xmax))`.
- Protected row/column budgets are `ceil(ratio * rows)` and
  `ceil(ratio * cols)`.
- The public BSGS bound is `M = ceil(2^sfp * xmax)^2 * K * R`.
- `S_P` receives only clear B rows and clear A columns as signed int64 factors.
- `S_D` receives only PC-DMCFE objects for the protected prefix and the public
  pivot candidate pool.
- The candidate pool contains at most `2*K*R` deterministic, spread-out rows
  and columns from the clear region. Complete pivoting over this public block
  requires every pivot to be nonzero in the protocol field and chooses the
  largest remaining real Schur-complement entry. This produces nested,
  nonsingular row/column choices while avoiding needlessly ill-conditioned
  skeleton cores. It is not used to estimate the rank of the complete
  plaintext block.
- Skeleton search starts at the configured LoRA rank `R` and increases one rank
  at a time, up to `K*R`. Previously decrypted entries are cached. Increasing
  the rank by one therefore adds only one C column and one S row, including
  `encrypted_B_rows + encrypted_A_cols` new bounded recoveries at `S_D`.
- For the paper's SEL-2S evaluation, the harness retains the complete quantized
  client factors in a separate plaintext oracle. These values are not placed in
  either server payload and are not counted as protocol communication.
- For every candidate rank, the oracle measures
  `||sum_i(B_i*A_i) - C*M^-1*S||_F / ||sum_i(B_i*A_i)||_F` in the encoded
  integer domain. The equivalent low-rank Gram computation avoids materializing
  the full `rows*cols` aggregate. The first rank at or below `1e-8` is returned.
- `S_D` decrypts only the protected cells required by C and S. M and the clear
  portions of C/S are computed by `S_P`. If no candidate rank passes, the round
  fails explicitly instead of returning a potentially incomplete aggregate.

The two servers are enforced as separate payloads and computation paths inside
one process. Physical process separation and transport serialization remain
future deployment work. Public pivot candidates must still contain a
nonsingular intersection large enough for the successful skeleton rank.

The current non-interactive implementation uploads a public reserve of at most
`2*K*R` clear-region pivot rows and `2*K*R` clear-region pivot columns to
`S_D` in encrypted form. Rank search then uses nested prefixes of the pivots
selected from that reserve. Replacing a singular pivot therefore causes no
additional upload, but every reserve label is included in client encryption
time and the measured `S_D` upload. This is deliberately more conservative
than the paper's final-rank expression `S_B + S_A + 2*R_W`.

## FULL+SK Data Flow

- Every B row and A column is encrypted with the same standard PC-DMCFE object
  shape; there is no plaintext helper and no compact zero encoding.
- A deterministic public `2*K*R` candidate block is decrypted to select nested
  nonsingular pivots. Candidate decryptions are measured separately and cached
  when they overlap the final skeleton.
- Rank search, fixed-point verification, C/M/S serialization, and output
  reparameterization are shared with `SEL-2S`.

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
    mode: str = "sel-2s",
)

session.encrypt_client(client_id, round_id, layers) -> NativeClientUpdate
session.aggregate_round(round_id, updates) -> list[NativeLayerSkeleton]
```

`NativeClientUpdate` exposes actual signed-int64 `S_P` bytes, actual mcl
`IoSerialize` `S_D` bytes, their total, label counts, and four client timing
stages. Each returned layer
exposes `layer_id`, `c`, `m`, and `s`, where C/M/S contain signed decoded
fixed-point integers. It also exposes:

- `selected_rank`: first rank in `R..K*R` whose baseline check passed.
- `baseline_checks`: number of attempted ranks.
- `baseline_relative_error`: encoded-domain relative Frobenius error at the
  selected rank.
- `decrypted_cells`: unique aggregate cells recovered with BSGS; in `FULL+SK`
  this includes candidate-pivot cells, while `pivot_candidate_cells` exposes
  that subset separately.

The native round metrics split `S_P` from `S_D`; the latter is further split
into DMCFE-mask aggregation, FE evaluation, BSGS search, and control logic.
`cur_skeleton_wall_sec` accumulates pivot elimination, protected-cell work-list
construction and assignment, every attempted CUR solve, and final C/M/S
materialization. Its diagnostic subset `cur_reconstruct_wall_sec` contains the
CUR solves and final C/M/S materialization only. Plaintext-oracle verification
is reported separately and excluded from the paper's protocol critical path.
Unattributed validation, dispatch, and bookkeeping time is reported as
`server_common_control_wall_sec` and remains in the critical path.

Three SecLoRA-specific files are written under each run's `metrics/` directory:

- `seclora_client.csv`: per-client quantization/packing, reusable precompute,
  online crypto, serialization, `S_P`/`S_D` upload bytes, and candidate labels.
- `seclora_round.csv`: independent server times, derived `max(S_P,S_D)` path,
  output reconstruction, real CMS download bytes, and all-client traffic.
- `seclora_layer.csv`: dimensions, selected rank, checks, decrypted cells, and
  C/M/S byte counts for every LoRA pair.

The paper-facing round fields are:

```text
FE aggregate = dFE mask aggregation + pairing FE evaluation
Decryption   = common control + max(S_P, S_D) + CUR skeleton
Server path  = Decryption + output reconstruction
System path  = max(client online) + Server path
Network(b)   = 8 * (mean upload/client + download/client) / b
```

`bsgs_wall_sec` is online bounded recovery only; reusable BSGS table
construction is excluded. `experiment_verify_wall_sec` is excluded from all
paper-facing paths. The measured single-process serial server wall time is
retained only as an audit value. All byte columns are raw bytes; tables convert
with `1 MiB = 2^20 bytes`.

### SecLoRA CSV fields

`seclora_client.csv` contains one row per client and round:

- identity: `round`, `client_id`, `mode`, `ratio`, `layer_count`;
- client stages: `quantize_pack_wall_sec`, `precompute_wall_sec`,
  `online_crypto_wall_sec`, `serialize_wall_sec`,
  `client_online_wall_sec`, and `client_total_crypto_wall_sec`;
- protected scope: `protected_b_labels`, `protected_a_labels`,
  `candidate_b_labels`, `candidate_a_labels`, and `encrypted_scalars`;
- actual wire sizes: `sp_upload_bytes`, `sd_upload_bytes`, and `upload_bytes`.

`client_online_wall_sec` is quantization/packing plus online crypto plus
serialization. `client_total_crypto_wall_sec` additionally includes reusable
precomputation.

`seclora_round.csv` contains one row per federated round:

- paper timing: `fe_aggregate_wall_sec`, `bsgs_wall_sec`,
  `cur_skeleton_wall_sec`, `decrypt_wall_sec`,
  `server_parallel_critical_wall_sec`, `system_critical_wall_sec`,
  `network_100mbps_wall_sec`, and `e2e_100mbps_wall_sec`;
- client summaries: `client_online_mean_wall_sec`,
  `client_online_max_wall_sec`, `client_total_crypto_mean_wall_sec`, and
  `encrypted_scalars_per_client_mean`;
- server audit: `sp_wall_sec`, `sd_wall_sec`, `sd_dfe_mask_wall_sec`,
  `sd_fe_eval_wall_sec`, `sd_bsgs_search_wall_sec`, `sd_control_wall_sec`,
  `cur_reconstruct_wall_sec`, `experiment_verify_wall_sec`,
  `server_common_control_wall_sec`, `output_reconstruct_wall_sec`, and
  `observed_serial_server_wall_sec`;
- upload: per-client mean `S_P`, `S_D`, and total bytes, diagnostic
  `upload_bytes_per_client_max`, and `upload_bytes_all_clients`;
- download: unique `C\\M`, `M`, and `S\\M` bytes, per-client/all-client totals,
  and `round_traffic_bytes_all_clients`;
- skeleton audit: `protected_skeleton_cells` and `pivot_candidate_cells`.

`seclora_layer.csv` records each LoRA pair's dimensions, selected rank,
baseline attempts/error, decrypted and candidate cells, and serialized C/M/S
part sizes. The per-layer C/M/S parts count the intersection M once.

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

After the smoke test passes, run one experiment or a batch:

```bash
python main.py --config configs/seclora_3b_sel2s_010.yaml
bash run_seclora_3b.sh end-to-end
bash run_seclora_3b.sh full-sk
bash run_seclora_3b.sh legacy-loss
bash run_seclora_3b.sh modern-loss
bash run_seclora_3b_pipeline.sh
```

Batch runs use the config stem as a run-id prefix and copy the exact YAML to
`output/<run-id>/experiment.yaml`. The pipeline runs the four SEL-2S ratios and
FULL+SK first, evaluates every resulting adapter on MMLU, and then runs only
the 100-round modern-loss configuration. It stores its run-id map, combined
log, and SecLoRA rolling-loss plot under
`output/seclora_3b_pipeline_<timestamp>/`. After a modern-loss run, plot its
rolling curve alone or beside other methods with repeatable `--series`
arguments:

```bash
python plot_seclora_loss.py \
  --series 'SecLoRA=output/<seclora-modern-run-id>' \
  --series 'CKKS+Skeleton=output/<ckks-modern-run-id>' \
  --output output/pdf/figure3_3b.pdf
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
