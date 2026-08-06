#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT="${1:-end-to-end}"

run_config() {
    local config="$1"
    echo "[SecLoRA batch] starting $config"
    RUN_NAME="${config%.yaml}" bash "$ROOT/run.sh" "$ROOT/configs/$config"
}

case "$EXPERIMENT" in
    end-to-end)
        run_config seclora_3b_sel2s_000125.yaml
        run_config seclora_3b_sel2s_001.yaml
        run_config seclora_3b_sel2s_010.yaml
        run_config seclora_3b_sel2s_025.yaml
        ;;
    full-sk)
        run_config seclora_3b_full_sk.yaml
        ;;
    legacy-loss)
        run_config seclora_loss_3b_legacy_001.yaml
        ;;
    modern-loss)
        run_config seclora_loss_3b_modern_001.yaml
        ;;
    all)
        "$0" end-to-end
        "$0" full-sk
        "$0" legacy-loss
        "$0" modern-loss
        ;;
    *)
        echo "usage: bash run_seclora_3b.sh {end-to-end|full-sk|legacy-loss|modern-loss|all}" >&2
        exit 2
        ;;
esac
