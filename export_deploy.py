"""Arma los dos archivos livianos que necesita la demo desplegada sin GPU.

model.pkl guarda el detector como pesos de PyTorch (yolo_pt_bytes), que en un free tier sin GPU
significaría instalar torch entero para poder cargarlos. Este script exporta esos pesos a ONNX
(corre con onnxruntime, mucho más chico) y separa el resto del bundle en un pickle aparte, para que
app.py no dependa de ultralytics/torch cuando corre desplegado.

Uso, después de que el notebook haya generado model.pkl:
    uv run --with onnx --with onnxruntime python export_deploy.py

Salida en deploy/: detector.onnx y pipeline.pkl. Esos dos son los que se suben a un release de
GitHub; ver publicar_release.sh.
"""
import shutil
import tempfile
from pathlib import Path

import joblib
from ultralytics import YOLO

RAIZ = Path(__file__).parent
MODEL_PKL = RAIZ / "model.pkl"
SALIDA = RAIZ / "deploy"


def main():
    if not MODEL_PKL.exists():
        raise SystemExit(f"No encuentro {MODEL_PKL}. Corré antes notebook_final.ipynb hasta el final.")

    bundle = joblib.load(MODEL_PKL)
    SALIDA.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        pesos_pt = Path(tmp) / "detector.pt"
        pesos_pt.write_bytes(bundle["yolo_pt_bytes"])
        print(f"Exportando a ONNX (imgsz={bundle['yolo_imgsz']})...")
        ruta_onnx = YOLO(str(pesos_pt)).export(format="onnx", imgsz=bundle["yolo_imgsz"], opset=12, simplify=True)
        shutil.copy(ruta_onnx, SALIDA / "detector.onnx")

    # todo lo demás del bundle (pipeline de prioridad, DBSCAN, umbral, ROI, metadatos) no depende de
    # torch para nada, así que se guarda tal cual en un pickle aparte, sin los pesos del detector
    liviano = {k: v for k, v in bundle.items() if k != "yolo_pt_bytes"}
    joblib.dump(liviano, SALIDA / "pipeline.pkl", compress=3)

    mb_onnx = (SALIDA / "detector.onnx").stat().st_size / 1e6
    mb_pipe = (SALIDA / "pipeline.pkl").stat().st_size / 1e6
    print(f"deploy/detector.onnx: {mb_onnx:.1f} MB")
    print(f"deploy/pipeline.pkl: {mb_pipe:.1f} MB")
    print("Listo. Siguiente paso: ./publicar_release.sh <tag>")


if __name__ == "__main__":
    main()
