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
DIR_FIGS = Path(__file__).parent / "figs"
MUESTRA_CSV = Path(__file__).parent / "muestra_demo.csv"
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


@st.cache_data
def cargar_muestra():
    return pd.read_csv(MUESTRA_CSV)


def agrupar_focos(det, cfg):
    """DBSCAN (haversine) sobre las detecciones de prioridad alta/media. Devuelve (con_foco, tabla_focos)."""
    urgentes = det[det["prioridad"].isin(["alta", "media"])].copy()
    if len(urgentes) < cfg["min_samples"]:
        return urgentes, None
    urgentes["foco"] = DBSCAN(eps=cfg["eps_m"] / 6_371_000, min_samples=cfg["min_samples"],
                              metric="haversine").fit_predict(np.radians(urgentes[["lat", "lon"]].values))
    con_foco = urgentes[urgentes["foco"] >= 0]
    tabla = con_foco.groupby("foco").agg(
        detecciones=("foco", "size"), altas=("prioridad", lambda s: int((s == "alta").sum())),
        barrio=("barrio", "first"), lat=("lat", "mean"), lon=("lon", "mean"))
    return urgentes, tabla


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

analizar, focos, roi, proyecto = st.tabs(["Analizar fotos", "Focos", "ROI", "El proyecto"])

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
    subidas = st.session_state.get("detecciones", pd.DataFrame())
    fuente = st.radio("Detecciones a agrupar", ["Datos de muestra (100 del test)", "Fotos que subiste"],
                      horizontal=True, index=0 if subidas.empty else 1)
    det = cargar_muestra() if fuente.startswith("Datos") else subidas
    cfg = bundle["dbscan"]
    st.caption(f"DBSCAN con distancia haversine, eps = {cfg['eps_m']} m, min_samples = {cfg['min_samples']}, "
               f"sobre las detecciones de prioridad alta o media.")

    if det.empty:
        st.info("Subí fotos en la pestaña \"Analizar fotos\" para usar tus propias detecciones acá.")
    else:
        urgentes, tabla_focos = agrupar_focos(det, cfg)
        if tabla_focos is None:
            st.info(f"Hacen falta al menos {cfg['min_samples']} detecciones de prioridad alta o media para "
                    f"formar un foco. Van {len(urgentes)}.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Focos", len(tabla_focos))
            c2.metric("Detecciones agrupadas", int((urgentes["foco"] >= 0).sum()))
            c3.metric("Ruido (sin foco)", int((urgentes["foco"] == -1).sum()))
            st.map(urgentes[urgentes["foco"] >= 0][["lat", "lon"]])
            st.dataframe(tabla_focos.sort_values(["altas", "detecciones"], ascending=False))

with roi:
    st.write("El mecanismo es simple: un bache detectado temprano se repara cuando todavía es chico, "
             "y eso ahorra asfalto.")
    st.latex(r"\text{ahorro} = \text{baches/año} \times \text{cobertura} \times \text{recall}_{D40} "
             r"\times \text{m}^2\text{/bache} \times \text{reducción de área} \times \text{costo/m}^2")
    st.markdown(
        f"- **baches/año**: 42.900 (3.574 bacheos mensuales × 12, dato de la Municipalidad de Asunción).\n"
        f"- **cobertura**: qué fracción de los baches de la ciudad llega a relevarse con este sistema. Supuesto propio.\n"
        f"- **recall D40**: **{bundle['metricas']['test_detector']['recall_D40']:.3f}**, medido en test — el único "
        f"número de esta fórmula que sale del modelo, no de un supuesto.\n"
        f"- **m²/bache**: 3,8 (6.600 m² reparados en 1.717 bacheos, dato municipal).\n"
        f"- **reducción de área**: cuánto más chico sale el parche por repararlo temprano en vez de tarde. Supuesto propio.\n"
        f"- **costo/m²**: hasta 305.000 Gs (cota superior, mezcla regularización asfáltica y bacheo). Supuesto propio.\n\n"
        f"Al ahorro se le resta el **costo anual del sistema** (server, mantenimiento) para llegar al beneficio neto; "
        f"`ROI = beneficio neto / costo del sistema`. Los tres escenarios cambian cobertura, reducción de área y "
        f"costo del sistema — cobertura y reducción de área todavía no están validadas con la Dirección de Vialidad."
    )
    st.dataframe(pd.DataFrame(bundle["roi"]).set_index("escenario"))

with proyecto:
    st.subheader("El problema")
    st.write("Asunción no tiene un catastro actualizado de daños viales: la inspección es manual y "
             "reactiva, se sale a reparar cuando entra un reclamo, así que los baches se atienden "
             "tarde, cuando ya crecieron y cuesta más repararlos. Esto es para la Dirección de "
             "Vialidad de la Municipalidad: necesitan saber dónde están los daños y cuáles atender "
             "primero para planificar las cuadrillas de bacheo, en vez de ir reclamo por reclamo.")

    st.subheader("Por qué Machine Learning / Deep Learning")
    st.write("Detectar daños en una foto de calle a ojo, una por una, no escala: no hay forma de que "
             "un inspector recorra toda la ciudad con la frecuencia que haría falta. Un detector "
             "entrenado (YOLOv8n) sí puede correr sobre miles de fotos, y la parte de prioridad y "
             "agrupamiento (`Pipeline` de scikit-learn + DBSCAN) convierte esas detecciones sueltas "
             "en algo que una cuadrilla puede usar para planificar.")

    st.subheader("Métrica de éxito")
    st.write("**Recall**, no accuracy. Un bache que el sistema no detecta sigue rompiendo autos; una "
             "falsa alarma sólo cuesta que un inspector mire una foto de más. Por eso el umbral de "
             "confianza se elige maximizando F2 (pondera el recall el doble que la precisión), y se "
             "reportan también precisión, F1 y mAP50.")

    sel = bundle["seleccion"]
    met = bundle["metricas"]
    n_img = sel["n_imagenes"]

    st.subheader("Arquitectura")
    st.graphviz_chart(f"""
        digraph {{
            rankdir=LR
            fontname="sans-serif"
            node [shape=box, style="rounded,filled", fillcolor="#f5f5f5", fontname="sans-serif", fontsize=11]
            edge [fontname="sans-serif", fontsize=9]

            subgraph cluster_0 {{
                label="Etapa 1 · Detección — YOLOv8n"; style=dashed; fontname="sans-serif"; fontsize=11
                img [label="Imagen\\n640×640×3"]
                backbone [label="Backbone + Neck\\nCSPDarknet / PAN-FPN"]
                head [label="Head anchor-free\\n3 escalas"]
                nms [label="NMS\\nconf ≥ {bundle['umbral_confianza']}"]
                img -> backbone -> head -> nms
            }}
            det [label="Detección\\n(clase, confianza, cx,cy,w,h)"]
            nms -> det

            subgraph cluster_1 {{
                label="Etapa 2 · Prioridad — scikit-learn"; style=dashed; fontname="sans-serif"; fontsize=11
                feat [label="9 features\\n6 numéricas + 3 categóricas"]
                ct [label="ColumnTransformer\\nimputer+scaler / imputer+onehot"]
                clf [label="{sel['prioridad']['modelo']}\\n(elegido por CV)"]
                feat -> ct -> clf
            }}
            det -> feat [label="  + barrio, vía, GPS"]
            prio [label="Prioridad\\nalta / media / baja"]
            clf -> prio

            subgraph cluster_2 {{
                label="Etapa 3 · Focos"; style=dashed; fontname="sans-serif"; fontsize=11
                dbscan [label="DBSCAN haversine\\neps={bundle['dbscan']['eps_m']} m, min_samples={bundle['dbscan']['min_samples']}"]
            }}
            prio -> dbscan [label="  alta/media"]
            focos [label="Focos geográficos"]
            dbscan -> focos
        }}
    """)

    st.subheader("Protocolo de datos y entrenamiento")
    st.markdown(
        f"- **Dataset**: RDD2022, split provisto por la fuente — {n_img['train']:,} train / "
        f"{n_img['val']:,} val / {n_img['test']:,} test. Se excluyó `{sel['paises_excluidos'][0]}` "
        f"(vista aérea, otro dominio que el de fotos de smartphone).\n"
        f"- **Inicialización**: `{sel['detector']['inicializacion']}`, decidido comparando contra pesos "
        f"aleatorios con el mismo presupuesto (tabla abajo).\n"
        f"- **Augmentation**: flip horizontal (p=0,5), HSV suave, mosaic apagado en las últimas "
        f"{sel['detector']['augmentation']['close_mosaic']} épocas.\n"
        f"- **Entrenamiento**: `imgsz={sel['detector']['imgsz']}`, hasta la época "
        f"{sel['detector']['mejor_epoca']}, {sel['detector']['minutos']:.0f} min.\n"
        f"- **Selección de umbral e hiperparámetros**: todo se decide en validación (`{sel['detector']['regla_umbral']}` "
        f"para el umbral, `{sel['prioridad']['regla']}` para el modelo de prioridad), se congela en "
        f"`selection.json` y el test se evalúa una sola vez."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Comparación de inicialización (mAP50 val, mismo presupuesto corto)")
        st.dataframe(pd.DataFrame(sel["detector"]["comparacion_transfer"]).T.round(4))
    with c2:
        st.caption("Selección del modelo de prioridad (CV agrupada por imagen)")
        st.dataframe(pd.DataFrame(sel["prioridad"]["cv"]).round(3))

    st.subheader("Métricas en test")
    c3, c4 = st.columns(2)
    c3.dataframe(pd.DataFrame([met["test_detector"]]).T.rename(columns={0: "detector"}).round(4))
    c4.dataframe(pd.DataFrame([met["test_prioridad"]]).T.rename(columns={0: "prioridad"}).round(4))
    st.caption("La georreferenciación y el target de prioridad son simulados en este proyecto — "
               "RDD2022 no tiene fotos de Asunción. El detalle está en el notebook y el README.")

    st.subheader("Resultados de la corrida real")
    c1, c2 = st.columns(2)
    c1.image(str(DIR_FIGS / "eda.png"), caption="Análisis exploratorio: balance de clases y tamaño de las cajas")
    c2.image(str(DIR_FIGS / "curvas_entrenamiento.png"), caption="Entrenamiento: pérdida y mAP50 por época")
    c3, c4 = st.columns(2)
    c3.image(str(DIR_FIGS / "umbral_validacion.png"), caption="Elección del umbral de confianza (máximo F2)")
    c4.image(str(DIR_FIGS / "prioridad_test.png"), caption="Matriz de confusión de la etapa de prioridad")
    c5, c6 = st.columns(2)
    c5.image(str(DIR_FIGS / "focos_dbscan.png"), caption="Focos de DBSCAN sobre los tramos simulados")
    c6.image(str(DIR_FIGS / "verificacion_clases.png"), caption="Verificación manual del mapeo de clases")

    st.subheader("Próximos pasos")
    st.markdown(
        "1. Juntar y etiquetar unas 500 fotos de Asunción con GPS, para medir el cambio de dominio y "
        "hacer fine-tuning local.\n"
        "2. Exportar el detector a TFLite INT8 y medir la latencia en Android, apuntando a 50 ms o "
        "menos con una caída de mAP de hasta 3 puntos.\n"
        "3. Reemplazar la regla de prioridad por etiquetas reales de inspectores."
    )
