"""
EcoVision - Punto de entrada principal
======================================
Ejecuta la interfaz de Streamlit para el reconocimiento de materiales.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Inicia la aplicación de Streamlit."""
    ruta_interfaz = Path(__file__).parent / "interfaz" / "app.py"

    if not ruta_interfaz.exists():
        print(f"Error: No se encontró {ruta_interfaz}")
        sys.exit(1)

    print("Iniciando EcoVision...")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ruta_interfaz)],
        check=True,
    )


if __name__ == "__main__":
    main()
