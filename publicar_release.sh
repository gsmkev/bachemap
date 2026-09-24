#!/usr/bin/env bash
# Sube deploy/detector.onnx y deploy/pipeline.pkl como assets de un release de GitHub.
# Uso: ./publicar_release.sh v1
set -euo pipefail

TAG="${1:?uso: ./publicar_release.sh <tag>, por ejemplo v1}"

if [ ! -f deploy/detector.onnx ] || [ ! -f deploy/pipeline.pkl ]; then
    echo "Falta deploy/detector.onnx o deploy/pipeline.pkl. Correr antes:"
    echo "  uv run --with onnx --with onnxruntime python export_deploy.py"
    exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
    echo "Hace falta la CLI de GitHub (gh) para este script: https://cli.github.com/"
    echo "Alternativa: crear el release '$TAG' a mano en GitHub y subir deploy/detector.onnx y deploy/pipeline.pkl como assets."
    exit 1
fi

gh release create "$TAG" deploy/detector.onnx deploy/pipeline.pkl \
    --title "Modelo $TAG" \
    --notes "Detector en ONNX y pipeline de prioridad, generados con export_deploy.py."

echo "Listo. Si el tag no es v1, actualizar BACHEMAP_MODEL_TAG en el deploy de Render."
