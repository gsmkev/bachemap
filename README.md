# BacheMap

Detección, priorización y georreferenciación de daños viales a partir de fotos de calle.

Proyecto final de Kevin Galeano y Werner Uibrig para el Diplomado en Machine Learning y Deep Learning
Aplicado, FIUNA 2026.

## El problema

Asunción no tiene un catastro actualizado de daños viales. La inspección es manual y reactiva: se sale a
reparar cuando entra un reclamo, así que los baches se atienden tarde, cuando ya crecieron y cuesta más
repararlos. La Dirección de Vialidad de la Municipalidad necesita saber dónde están los daños y cuáles
atender primero para planificar las cuadrillas de bacheo.

## Objetivo

Dada una foto de calle tomada con un celular, detectar los daños, asignarle a cada uno una prioridad de
reparación y agrupar los daños cercanos en focos, para que una cuadrilla pueda salir a cubrir una zona en
vez de un reclamo suelto.

**Métrica de éxito: recall**, no accuracy. Un bache que el sistema no detecta sigue rompiendo autos; una
falsa alarma sólo cuesta que un inspector mire una foto de más. Por eso el umbral de confianza se elige
maximizando F2, que pondera el recall el doble que la precisión. Se reportan también precisión, F1 y mAP50.

## Datos

[RDD2022](https://github.com/sekilab/RoadDamageDetector) (Arya et al., 2022; CC BY-SA 4.0), en el mirror de
Kaggle [`aliabdelmenam/rdd-2022`](https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022), que viene en
formato YOLO con las particiones train/val/test ya hechas.

Son 38.385 imágenes de calles de seis países con 55.005 cajas etiquetadas en cuatro clases:

| Clase | Daño | Proporción |
|---|---|---|
| D00 | Grieta longitudinal | 47,3 % |
| D10 | Grieta transversal | 21,5 % |
| D20 | Piel de cocodrilo | 19,3 % |
| D40 | Bache | 11,9 % |

La clase que más importa para el caso de uso, el bache, es la más escasa: por eso se mira el recall por
clase y no un promedio global. Se excluyen las 2.401 imágenes de China_Drone, que son tomas aéreas y no
fotos de smartphone.

Un detalle del dataset que vale la pena mirar: los ids de clase de este mirror no siguen el orden habitual
(el 3 es `other_corruption`, que se descarta, y el 4 es `pothole`). El notebook verifica el mapeo mostrando
recortes de cajas de cada clase, en `figs/verificacion_clases.png`.

## El modelo

Dos etapas, porque el problema tiene dos preguntas distintas:

```
foto → [1. YOLOv8n] → detecciones (clase, confianza, caja)
                        + contexto (barrio, tipo de vía, GPS)
                            ↓
       [2. Pipeline sklearn: ColumnTransformer + clasificador] → prioridad alta/media/baja
                            ↓
       [DBSCAN con distancia haversine] → focos para planificar cuadrillas
```

1. **Detección.** YOLOv8n a 640 px, partiendo de los pesos preentrenados en COCO. El notebook compara esa
   inicialización contra entrenar desde cero con el mismo presupuesto, y la decisión la toma el mAP50 de
   validación. El umbral de confianza también sale de validación, por F2.
2. **Prioridad.** Cada detección se vuelve una fila de una tabla y pasa por un `Pipeline` con
   `ColumnTransformer` (imputación y escalado para las numéricas, one-hot para las categóricas) y un
   clasificador. Se comparan regresión logística y random forest con validación cruzada agrupada por
   imagen, para que las detecciones de una misma foto no queden repartidas entre folds.
3. **Focos.** DBSCAN sobre las coordenadas de las detecciones urgentes, con distancia haversine y eps de
   100 m. No hace falta fijar de antemano cuántos focos hay, y los daños aislados quedan como ruido.

El protocolo: la validación decide todo, las decisiones quedan escritas en `artefactos/selection.json`
antes de tocar el test, y el test se evalúa una sola vez. Lo que salga ahí se reporta, no se usa para
volver a ajustar.

## Resultados

Detector, en test, con el umbral elegido en validación:

| Métrica | Valor |
|---|---|
| mAP50 | 0,548 |
| mAP50-95 | 0,278 |
| Precisión | 0,424 |
| Recall | 0,618 |
| Recall de baches (D40) | 0,509 |

Prioridad, en test: recall de `alta` 0,889, F2 de `alta` 0,865, F1 macro 0,789.

Focos: 21 focos detectados, ARI 0,731 contra los tramos simulados.

ROI anual estimado, en millones de guaraníes:

| Escenario | Ahorro | Beneficio neto | ROI |
|---|---|---|---|
| Pesimista | 251,5 | 1,5 | 0,01 |
| Base | 1.676,6 | 1.476,6 | 7,38 |
| Optimista | 4.602,3 | 4.452,3 | 29,68 |

Los valores salen de `resumen_resultados.json`, que genera el notebook.

El cálculo del ahorro usa el recall de baches medido en test, cifras publicadas por la Municipalidad
(3.574 bacheos en agosto de 2026, 3,8 m² por bache, unos 305.000 Gs/m²) y tres supuestos propios que hay
que validar con Vialidad: cobertura del relevamiento, cuánto se achica el área a reparar por detectar
temprano, y el costo de operar el sistema.

## Lo que está simulado

RDD2022 no tiene fotos de Asunción, así que dos cosas del proyecto son simuladas y se declaran como tales:

- **La georreferenciación.** Cada imagen se asigna a uno de 30 tramos simulados repartidos en 10 barrios
  reales de Asunción. En producción, lat/lon salen del GPS del teléfono y el tipo de vía de la cartografía
  municipal.
- **El target de prioridad.** Es una regla operativa declarada: falsa alarma → `baja`; si no, puntaje por
  severidad de la clase real, tamaño de la caja y si la vía es arterial, con un 5 % de desacuerdo simulado
  entre inspectores. El modelo no ve la clase real ni si la detección es falsa alarma: tiene que inferirlas
  de la clase predicha, la confianza, el tamaño y el contexto.

Consecuencia: las métricas de la etapa 2 salen altas porque el modelo recupera una regla que escribimos
nosotros. No es validación del criterio de un inspector real. Lo mismo con el ARI de los focos: valida que
el mecanismo agrupa bien, no que funcione en calles reales.

## Cómo ejecutarlo

Se instala y se corre con [uv](https://docs.astral.sh/uv/). Requiere una GPU NVIDIA con CUDA: el
entrenamiento completo tarda unas 3 o 4 horas.

```bash
uv sync                 # crea el entorno con las versiones exactas de uv.lock
cp .env.example .env    # y completar las claves (ver abajo)

uv run jupyter lab notebook_final.ipynb     # ejecutar de arriba a abajo
uv run streamlit run app.py                 # la demo, después de generar model.pkl
```

El notebook baja el dataset con `kagglehub` (unos 12 GB, cacheados en `~/.cache/kagglehub`, así que la
descarga ocurre una sola vez), entrena, evalúa y deja todo en `bachemap_work/`. Al terminar copia
`model.pkl` y `datos_muestra.csv` a esta carpeta.

### Demo desplegada sin GPU

`app.py` no usa `model.pkl` directamente: el detector corre con `onnxruntime` en vez de ultralytics/torch,
para poder desplegarlo en un free tier sin GPU (por ejemplo Render). El flujo es:

```bash
uv run --with onnx --with onnxruntime python export_deploy.py   # model.pkl -> deploy/detector.onnx + deploy/pipeline.pkl
./publicar_release.sh v1                                        # sube esos dos archivos a un release de GitHub (necesita gh)
uv run streamlit run app.py                                      # los baja de ahí la primera vez y los cachea en deploy/
```

En Render: build command `pip install -r requirements-app.txt`, start command
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0` (ya está en `render.yaml`). La variable
`BACHEMAP_MODEL_TAG` elige qué release de modelos usar (por defecto `v1`).

### Variables de entorno (`.env`)

| Variable | Para qué |
|---|---|
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Bajar el dataset. Salen del `kaggle.json` de *Account → Settings → API → Create New API Token*. |
| `WANDB_API_KEY` | Registrar la corrida en Weights & Biases. Está en <https://wandb.ai/authorize>. Sin esta variable el notebook corre igual, sin trazabilidad. |
| `WANDB_PROJECT` | Nombre del proyecto en W&B (por defecto `bachemap`). |
| `BACHEMAP_WANDB=0` | Apagar W&B aunque haya key. |
| `BACHEMAP_INPUT` | Ruta a un dataset ya descargado: saltea la descarga. |
| `BACHEMAP_WORK` | Dónde escribir los artefactos (por defecto `./bachemap_work`). |
| `BACHEMAP_PRUEBA=1` | Corrida corta en CPU para probar que el código anda, sin entrenar de verdad. |

El `.env` está en `.gitignore`; el que se versiona es `.env.example`.

### Trazabilidad con Weights & Biases

Si hay `WANDB_API_KEY`, el notebook abre una sola corrida al principio y todo queda ahí dentro:

- la configuración del experimento (modelo base, `imgsz`, batch, épocas, semilla, dataset);
- las curvas por época del entrenamiento y de la comparación de inicialización, que registra la
  integración nativa de Ultralytics (reutiliza la corrida ya abierta, así que no se parte en varias);
- al cerrar: las métricas de test como resumen;
- un artefacto `bachemap-model` con `model.pkl`, para saber siempre qué pesos corresponden a qué corrida.

La URL de la corrida se imprime en la sección 1.

### Ajustes según la máquina

Todo en la celda 0: `BATCH` (16 entra en 6 GB de VRAM; bajarlo si aparece `CUDA out of memory`),
`WORKERS`, `EPOCHS` y `MAX_IMAGENES`.

En Linux x86_64 el `torch` de PyPI ya viene con CUDA 12 y `uv sync` alcanza. Para fijar otra versión de
CUDA, el `pyproject.toml` tiene comentado el índice de PyTorch.

## Estructura

```
bachemap/
├── notebook_final.ipynb   # todo el pipeline, de la descarga del dataset al model.pkl
├── app.py                 # demo en Streamlit: analizar fotos, focos, ROI
├── onnx_infer.py           # pre/post-procesamiento de YOLO para onnxruntime, sin ultralytics/torch
├── export_deploy.py        # model.pkl -> deploy/detector.onnx + deploy/pipeline.pkl
├── publicar_release.sh     # sube esos dos archivos a un release de GitHub
├── requirements-app.txt    # dependencias mínimas para desplegar app.py sin GPU
├── render.yaml              # config de despliegue en Render
├── model.pkl               # lo genera el notebook
├── datos_muestra.csv       # lo genera el notebook: 100 detecciones del test
├── pyproject.toml          # dependencias
├── uv.lock                 # versiones exactas
├── .env.example            # plantilla de credenciales
└── README.md
```

`model.pkl` es un diccionario de objetos estándar: los bytes de los pesos de YOLO, el `Pipeline` de
scikit-learn, los parámetros de DBSCAN y los metadatos. Sin clases propias, así que `joblib.load()`
funciona sin importar código de este proyecto. `export_deploy.py` separa ese mismo contenido en dos
archivos livianos para el despliegue: el detector convertido a ONNX y el resto del bundle (sin los
pesos de YOLO) en `pipeline.pkl`.

## Límites

- Una sola semilla y un solo presupuesto de entrenamiento: la diferencia entre configuraciones parecidas no
  es concluyente.
- RDD2022 no incluye calles de Asunción. Antes de desplegar hay que medir el rendimiento con fotos locales.
- La georreferenciación y el target de prioridad son simulados, como se explicó arriba.
- El ROI depende de supuestos propios que todavía no están validados con Vialidad.

## Próximos pasos

1. Juntar y etiquetar unas 500 fotos de Asunción con GPS, para medir el cambio de dominio y hacer
   fine-tuning local.
2. Exportar el detector a TFLite INT8 y medir la latencia en Android, apuntando a 50 ms o menos con una
   caída de mAP de hasta 3 puntos.
3. Reemplazar la regla de prioridad por etiquetas reales de inspectores.

## Armar el ZIP de la entrega

```bash
zip galeano_uibrig_bachemap.zip README.md notebook_final.ipynb model.pkl datos_muestra.csv app.py onnx_infer.py
du -h galeano_uibrig_bachemap.zip     # tiene que pesar menos de 20 MB
```

El notebook que va al ZIP es el ya ejecutado, con sus salidas.
