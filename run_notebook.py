"""Ejecuta notebook_final.ipynb de punta a punta, guardándolo después de cada celda.

`jupyter nbconvert --execute --inplace` no escribe nada hasta el final, así que si una celda tira
una excepción a mitad de camino (nos pasó con un fallo de W&B después de que el entrenamiento y la
evaluación ya habían terminado bien) se pierde todo el trabajo de una corrida de varias horas. Este
script guarda el notebook a disco después de cada celda, para poder ver el progreso mientras corre y
no perder nada si se cae en cualquier punto.

Uso: uv run python run_notebook.py
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RUTA = Path(__file__).parent / "notebook_final.ipynb"


def main():
    nb = nbformat.read(RUTA, as_version=4)
    cliente = NotebookClient(nb, timeout=-1, kernel_name="python3")

    def guardar(**_kwargs):   # se registra para on_cell_executed y on_cell_error: guarda pase lo que pase
        nbformat.write(nb, RUTA)

    cliente.on_cell_executed = guardar
    cliente.on_cell_error = guardar

    fallo = None
    try:
        cliente.execute()
    except Exception as e:   # cualquier motivo (celda con error, kernel muerto, lo que sea): ya quedó guardado
        fallo = e
    finally:
        nbformat.write(nb, RUTA)   # por si el fallo fue entre celdas (p. ej. el kernel murió)

    if fallo:
        print(f"Terminó con una celda en error: {fallo}", file=sys.stderr)
        sys.exit(1)
    print("Notebook ejecutado de punta a punta sin errores.")


if __name__ == "__main__":
    main()
