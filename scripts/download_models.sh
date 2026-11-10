#!/bin/sh
# Download the models into $1 (default ./models).
#   RAM++ ONNX       - Apache-2.0
#   InsightFace antelopev2 - NON-COMMERCIAL RESEARCH licence (https://github.com/deepinsight/insightface)
set -eu
DEST="${1:-./models}"
mkdir -p "$DEST/ram-plus" "$DEST/insightface/models"

RAM=https://huggingface.co/imagent/ram-plus-onnx/resolve/main
for f in ram_plus.onnx ram_tag_list.txt ram_tag_list_threshold.txt LICENSE; do
  [ -f "$DEST/ram-plus/$f" ] || curl -fL --retry 3 -o "$DEST/ram-plus/$f" "$RAM/$f"
done
echo "4197a2cd19b6623e1286e9feda991e793fed7749ed9cee9793b174fee0eb844f  $DEST/ram-plus/ram_plus.onnx" | sha256sum -c -

if [ ! -f "$DEST/insightface/models/antelopev2/glintr100.onnx" ]; then
  echo "InsightFace antelopev2 is licensed for non-commercial research use only."
  curl -fL --retry 3 -o /tmp/antelopev2.zip https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip
  mkdir -p "$DEST/insightface/models/antelopev2"
  unzip -o -j /tmp/antelopev2.zip '*.onnx' -d "$DEST/insightface/models/antelopev2"   # zip nests a folder
  rm -f /tmp/antelopev2.zip
fi
ls -la "$DEST/ram-plus" "$DEST/insightface/models/antelopev2"
