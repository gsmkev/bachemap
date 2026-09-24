"""BacheMap: app de demostración.

Detecta daños en las fotos que se suban, les asigna una prioridad de reparación, agrupa las
detecciones en focos y muestra el ROI estimado.

Dos formas de conseguir el detector, según qué haya a mano:
- Si está `model.pkl` (lo genera notebook_final.ipynb) se usa ese, con ultralytics/torch directo.
  Es el camino natural para probar en la propia máquina, recién entrenado.
- Si no está, se bajan `detector.onnx` y `pipeline.pkl` de un release de GitHub y se corre con
  onnxruntime, sin torch. Es el camino para un deploy sin GPU (Render, por ejemplo). Para generar
  esos dos archivos a mano: export_deploy.py.

    streamlit run app.py
"""
import os
import tempfile
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ExifTags
from sklearn.cluster import DBSCAN

from onnx_infer import predecir

REPO_GITHUB = "gsmkev/bachemap"
TAG_MODELO = os.environ.get("BACHEMAP_MODEL_TAG", "v1")
MODEL_PKL = Path(__file__).parent / "model.pkl"
DIR_MODELOS = Path(__file__).parent / "deploy"
COLOR_PRIORIDAD = {"alta": (220, 30, 30), "media": (240, 160, 0), "baja": (90, 160, 90)}

st.set_page_config(page_title="BacheMap", layout="wide")


def asegurar_modelos():
    """Baja detector.onnx y pipeline.pkl del release de GitHub si todavía no están en deploy/."""
    DIR_MODELOS.mkdir(exist_ok=True)
    for archivo in ["detector.onnx", "pipeline.pkl"]:
        destino = DIR_MODELOS / archivo
        if destino.exists():
            continue
        url = f"https://github.com/{REPO_GITHUB}/releases/download/{TAG_MODELO}/{archivo}"
        with st.spinner(f"Bajando {archivo} ({TAG_MODELO})..."):
            try:
                urllib.request.urlretrieve(url, destino)
            except Exception as e:
                st.error(f"No pude bajar {archivo} de {url}: {e}")
                st.stop()


@st.cache_resource
def cargar_modelos():
    if MODEL_PKL.exists():
        from ultralytics import YOLO   # sólo se importa acá: no hace falta torch en el deploy sin GPU
        bundle = joblib.load(MODEL_PKL)
        pesos = Path(tempfile.gettempdir()) / "bachemap_yolo.pt"
        pesos.write_bytes(bundle["yolo_pt_bytes"])
        return bundle, YOLO(str(pesos)), "yolo"

    asegurar_modelos()
    import onnxruntime as ort
    bundle = joblib.load(DIR_MODELOS / "pipeline.pkl")
    sesion = ort.InferenceSession(str(DIR_MODELOS / "detector.onnx"), providers=["CPUExecutionProvider"])
    return bundle, sesion, "onnx"


def detectar_crudo(modelo, backend, bundle, imagen):
    """Detecciones sin procesar: lista de (clase_idx, confianza, cx, cy, w, h) normalizados."""
    if backend == "yolo":
        r = modelo.predict(imagen, conf=bundle["umbral_confianza"], imgsz=bundle["yolo_imgsz"], verbose=False)[0]
        cajas = r.boxes
        return [(int(k), float(c), float(cx), float(cy), float(w), float(h))
                for (cx, cy, w, h), c, k in zip(cajas.xywhn.cpu().numpy(), cajas.conf.cpu().numpy(),
                                                cajas.cls.cpu().numpy().astype(int))]
    dets = predecir(modelo, imagen, bundle["yolo_imgsz"], bundle["umbral_confianza"])
    return [(d["clase"], d["confianza"], d["cx"], d["cy"], d["w"], d["h"]) for d in dets]


def gps_del_exif(imagen):
    """(lat, lon) de los metadatos EXIF, o None si la foto no los trae."""
    exif = imagen.getexif()
    gps = exif.get_ifd(ExifTags.IFD.GPSInfo) if exif else None
    if not gps or 2 not in gps or 4 not in gps:
        return None

    def a_grados(valor, ref):
        g, m, s = (float(x) for x in valor)
        dec = g + m / 60 + s / 3600
        return -dec if ref in ("S", "W") else dec

    return a_grados(gps[2], gps.get(1, "N")), a_grados(gps[4], gps.get(3, "E"))


def barrio_mas_cercano(barrios, lat, lon):
    return min(barrios, key=lambda b: (barrios[b][0] - lat) ** 2 + (barrios[b][1] - lon) ** 2)


def detectar(modelo, backend, bundle, imagen, barrio, tipo_via):
    """Corre el detector y arma una fila por detección, con las features del pipeline de prioridad."""
    filas = []
    for k, conf, cx, cy, w, h in detectar_crudo(modelo, backend, bundle, imagen):
        filas.append({"clase_pred": bundle["clases"][k], "confianza": conf,
                      "area_rel": w * h, "aspecto": w / max(h, 1e-6),
                      "cy_rel": cy, "cx_rel": cx, "barrio": barrio, "tipo_via": tipo_via,
                      "cx": cx, "cy": cy, "w": w, "h": h})
    det = pd.DataFrame(filas)
    if det.empty:
        return det
    det["n_det_imagen"] = len(det)
    det["prioridad"] = bundle["pipeline_prioridad"].predict(det[bundle["features_num"] + bundle["features_cat"]])
    return det


def dibujar(imagen, det):
    lienzo = imagen.convert("RGB").copy()
    dibujo = ImageDraw.Draw(lienzo)
    W, H = lienzo.size
    for d in det.itertuples():
        caja = [(d.cx - d.w / 2) * W, (d.cy - d.h / 2) * H, (d.cx + d.w / 2) * W, (d.cy + d.h / 2) * H]
        color = COLOR_PRIORIDAD[d.prioridad]
        dibujo.rectangle(caja, outline=color, width=max(2, W // 400))
        dibujo.text((caja[0] + 4, caja[1] + 4), f"{d.clase_pred} {d.prioridad} {d.confianza:.2f}", fill=color)
    return lienzo


bundle, modelo, backend = cargar_modelos()
st.title("BacheMap")
st.caption(f"{bundle['nombre']} ({backend}) · umbral de confianza {bundle['umbral_confianza']} · "
           f"recall de baches en test {bundle['metricas']['test_detector']['recall_D40']:.2f}")

with st.sidebar:
    st.header("Contexto de la foto")
    st.write("En producción salen del GPS del teléfono y de la cartografía municipal. Acá se eligen a mano "
             "(si la foto trae GPS en el EXIF, se usa ese).")
    barrio_manual = st.selectbox("Barrio", sorted(bundle["barrios"]))
    tipo_via = st.selectbox("Tipo de vía", bundle["tipos_via"])

analizar, focos, roi = st.tabs(["Analizar fotos", "Focos", "ROI"])

with analizar:
    fotos = st.file_uploader("Fotos de calle", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    detecciones = []
    for foto in fotos or []:
        imagen = Image.open(foto)
        coords = gps_del_exif(imagen)
        barrio = barrio_mas_cercano(bundle["barrios"], *coords) if coords else barrio_manual
        lat, lon = coords if coords else bundle["barrios"][barrio]
        det = detectar(modelo, backend, bundle, imagen, barrio, tipo_via)

        izq, der = st.columns([2, 1])
        izq.image(dibujar(imagen, det) if not det.empty else imagen, caption=foto.name, use_container_width=True)
        der.write(f"**{barrio}** · {'GPS del EXIF' if coords else 'barrio elegido a mano'}")
        if det.empty:
            der.info("Sin detecciones por encima del umbral.")
        else:
            der.dataframe(det[["clase_pred", "confianza", "prioridad"]].round(3), hide_index=True)
            det["lat"], det["lon"] = lat, lon
            detecciones.append(det)

    st.session_state["detecciones"] = pd.concat(detecciones) if detecciones else pd.DataFrame()

with focos:
    det = st.session_state.get("detecciones", pd.DataFrame())
    urgentes = det[det["prioridad"].isin(["alta", "media"])] if not det.empty else det
    cfg = bundle["dbscan"]
    if len(urgentes) < cfg["min_samples"]:
        st.info(f"Hacen falta al menos {cfg['min_samples']} detecciones de prioridad alta o media para formar un foco. "
                f"Van {len(urgentes)}.")
    else:
        urgentes = urgentes.copy()
        urgentes["foco"] = DBSCAN(eps=cfg["eps_m"] / 6_371_000, min_samples=cfg["min_samples"],
                                  metric="haversine").fit_predict(np.radians(urgentes[["lat", "lon"]].values))
        st.map(urgentes[urgentes["foco"] >= 0][["lat", "lon"]])
        st.dataframe(urgentes[urgentes["foco"] >= 0].groupby("foco").agg(
            detecciones=("foco", "size"), altas=("prioridad", lambda s: int((s == "alta").sum())),
            barrio=("barrio", "first"), lat=("lat", "mean"), lon=("lon", "mean")))

with roi:
    st.write("Escenarios calculados en el notebook con el recall de baches medido en test. "
             "Los montos están en millones de guaraníes por año.")
    st.dataframe(pd.DataFrame(bundle["roi"]).set_index("escenario"))
