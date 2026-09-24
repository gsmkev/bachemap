"""Inferencia de YOLOv8 con onnxruntime, sin ultralytics ni torch.

Para el detector no hace falta más que esto: onnxruntime más un poco de numpy para el letterbox,
el reordenamiento de la salida y el NMS. La demo desplegada usa este módulo en vez de la librería
completa de Ultralytics para no tener que instalar torch en un servidor sin GPU.
"""
import numpy as np
from PIL import Image


def letterbox(imagen, tamano):
    """Reescala conservando la relación de aspecto y rellena con gris hasta (tamano, tamano)."""
    w, h = imagen.size
    escala = min(tamano / w, tamano / h)
    nw, nh = round(w * escala), round(h * escala)
    lienzo = Image.new("RGB", (tamano, tamano), (114, 114, 114))
    pad_x, pad_y = (tamano - nw) // 2, (tamano - nh) // 2
    lienzo.paste(imagen.convert("RGB").resize((nw, nh), Image.BILINEAR), (pad_x, pad_y))
    return lienzo, escala, (pad_x, pad_y)


def nms(cajas_xyxy, puntajes, iou_umbral):
    """NMS de siempre: por puntaje descendente, sin depender de opencv."""
    orden = puntajes.argsort()[::-1]
    elegidas = []
    while len(orden):
        i = orden[0]
        elegidas.append(i)
        if len(orden) == 1:
            break
        resto = orden[1:]
        x1 = np.maximum(cajas_xyxy[i, 0], cajas_xyxy[resto, 0])
        y1 = np.maximum(cajas_xyxy[i, 1], cajas_xyxy[resto, 1])
        x2 = np.minimum(cajas_xyxy[i, 2], cajas_xyxy[resto, 2])
        y2 = np.minimum(cajas_xyxy[i, 3], cajas_xyxy[resto, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_i = (cajas_xyxy[i, 2] - cajas_xyxy[i, 0]) * (cajas_xyxy[i, 3] - cajas_xyxy[i, 1])
        area_resto = (cajas_xyxy[resto, 2] - cajas_xyxy[resto, 0]) * (cajas_xyxy[resto, 3] - cajas_xyxy[resto, 1])
        iou = inter / (area_i + area_resto - inter + 1e-9)
        orden = resto[iou <= iou_umbral]
    return elegidas


def predecir(sesion, imagen, imgsz, conf_min, iou_nms=0.5):
    """Detecta cajas en la imagen. Devuelve una lista de dicts (clase, confianza, cx/cy/w/h normalizados)."""
    lienzo, escala, (pad_x, pad_y) = letterbox(imagen, imgsz)
    entrada = (np.asarray(lienzo, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]

    salida = sesion.run(None, {sesion.get_inputs()[0].name: entrada})[0][0]   # (4+n_clases, n_cajas)
    cajas_px, puntajes = salida[:4].T, salida[4:].T   # -> (n_cajas, 4) y (n_cajas, n_clases)
    clase = puntajes.argmax(1)
    confianza = puntajes.max(1)

    sobre_umbral = confianza >= conf_min
    cajas_px, clase, confianza = cajas_px[sobre_umbral], clase[sobre_umbral], confianza[sobre_umbral]
    if len(cajas_px) == 0:
        return []

    cx, cy, w, h = cajas_px.T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    ancho_orig, alto_orig = imagen.size

    detecciones = []
    for c in np.unique(clase):
        de_esta_clase = np.flatnonzero(clase == c)
        for i in nms(xyxy[de_esta_clase], confianza[de_esta_clase], iou_nms):
            idx = de_esta_clase[i]
            x1, x2 = (xyxy[idx, 0] - pad_x) / escala, (xyxy[idx, 2] - pad_x) / escala
            y1, y2 = (xyxy[idx, 1] - pad_y) / escala, (xyxy[idx, 3] - pad_y) / escala
            detecciones.append({
                "clase": int(c), "confianza": float(confianza[idx]),
                "cx": float((x1 + x2) / 2 / ancho_orig), "cy": float((y1 + y2) / 2 / alto_orig),
                "w": float((x2 - x1) / ancho_orig), "h": float((y2 - y1) / alto_orig),
            })
    return detecciones
