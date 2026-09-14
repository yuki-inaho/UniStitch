#!/usr/bin/env bash
# Download the fine-tuned ALIKED checkpoints from the GitHub release
# (https://github.com/yuki-inaho/UniStitch/releases/tag/v0.1.0-aliked).
#
#   pixi run download-models            # release assets (ALIKED, zero-padded 256-d)
#   pixi run download-models --with-hf  # + the released SuperPoint model from Hugging Face
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/model_homo_stage2"
BASE="https://github.com/yuki-inaho/UniStitch/releases/download/v0.1.0-aliked"

declare -A ASSETS=(
    [unistitch-aliked-zeropad-epoch0.pth]="db577eabb2f6134151a61ccb52a48ca939827e37ae7466c6434bc6d664112df9"
    [unistitch-aliked-zeropad-epoch2-ssim0.8649.pth]="216535781d880cf55dfc789934d3afaf3c4844286363ed3db31ca0f3cf3b852d"
)

mkdir -p "$DEST"

for name in "${!ASSETS[@]}"; do
    file="$DEST/$name"
    want="${ASSETS[$name]}"
    if [ -f "$file" ] && [ "$(sha256sum "$file" | cut -d' ' -f1)" = "$want" ]; then
        echo "==> $name already present and verified"
        continue
    fi
    echo "==> downloading $name"
    curl -fL --retry 3 --continue-at - -o "$file" "$BASE/$name"
    echo "$want  $file" | sha256sum -c -
done

if [ "${1:-}" = "--with-hf" ]; then
    hf_file="$DEST/epoch_best_model.pth"
    if [ -f "$hf_file" ]; then
        echo "==> epoch_best_model.pth already present"
    else
        echo "==> downloading epoch_best_model.pth from Hugging Face"
        curl -fL --retry 3 --continue-at - -o "$hf_file" \
            "https://huggingface.co/Y5Y/UniStitch_model/resolve/main/epoch_best_model.pth"
    fi
fi

echo "==> checkpoints ready in $DEST"
ls -la "$DEST"
