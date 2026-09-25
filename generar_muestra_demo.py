"""Genera muestra_demo.csv: una muestra más densa que datos_muestra.csv (que tiene el tope de 100
filas de la entrega), para que la pestaña Focos de la demo tenga suficientes puntos como para que
DBSCAN forme focos de verdad. No es parte de la entrega, sólo de la app.

Uso, con model.pkl en la raíz:
    uv run python generar_muestra_demo.py
"""
import random
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO

SEED = 123
N_IMAGENES = 1500
DIR_TEST = Path.home() / ".cache/kagglehub/datasets/aliabdelmenam/rdd-2022/versions/1/RDD_SPLIT/test/images"


def main():
    bundle = joblib.load("model.pkl")
    pesos = Path(tempfile.gettempdir()) / "bachemap_yolo_demo.pt"
    pesos.write_bytes(bundle["yolo_pt_bytes"])
    modelo = YOLO(str(pesos))

    rutas = sorted(DIR_TEST.glob("*.jpg"))
    random.seed(SEED)
    rutas = random.sample(rutas, min(N_IMAGENES, len(rutas)))
    print(f"{len(rutas)} imágenes de test")

    filas = []
    for i in range(0, len(rutas), 64):
        lote = rutas[i:i + 64]
        for r in modelo.predict(lote, conf=bundle["umbral_confianza"], imgsz=bundle["yolo_imgsz"], verbose=False):
            cajas = r.boxes
            n = len(cajas)
            for (cx, cy, w, h), conf, k in zip(cajas.xywhn.cpu().numpy(), cajas.conf.cpu().numpy(),
                                               cajas.cls.cpu().numpy().astype(int)):
                filas.append({"clase_pred": bundle["clases"][int(k)], "confianza": float(conf),
                             "area_rel": float(w * h), "aspecto": float(w / max(h, 1e-6)),
                             "cy_rel": float(cy), "cx_rel": float(cx), "n_det_imagen": n})
        print(f"  {min(i + 64, len(rutas))}/{len(rutas)}", flush=True)

    det = pd.DataFrame(filas)
    print(f"{len(det)} detecciones")

    # georreferenciación simulada, igual de espíritu que la del notebook pero sin tramos: alcanza
    # con dispersión alrededor del centro de cada barrio para que DBSCAN tenga algo que agrupar
    rng = np.random.default_rng(SEED)
    barrios = bundle["barrios"]
    elegido = rng.choice(list(barrios), len(det))
    det["barrio"] = elegido
    det["tipo_via"] = rng.choice(bundle["tipos_via"], len(det), p=[0.3, 0.3, 0.4])
    det["lat"] = [barrios[b][0] for b in elegido] + rng.normal(0, 0.003, len(det))
    det["lon"] = [barrios[b][1] for b in elegido] + rng.normal(0, 0.003, len(det))
    det["pais"] = "demo"
    det["img_id"] = [f"demo_{i}" for i in range(len(det))]

    det["prioridad"] = bundle["pipeline_prioridad"].predict(det[bundle["features_num"] + bundle["features_cat"]])

    columnas = ["img_id", "pais", "clase_pred", "tipo_via", "barrio", "confianza", "area_rel",
               "aspecto", "cy_rel", "cx_rel", "n_det_imagen", "lat", "lon", "prioridad"]
    det[columnas].round(5).to_csv("muestra_demo.csv", index=False)
    print("muestra_demo.csv:", det.shape)
    print(det["prioridad"].value_counts().to_dict())


if __name__ == "__main__":
    main()
