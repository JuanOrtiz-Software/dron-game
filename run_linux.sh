#!/bin/bash
cd "$(dirname "$0")" || {
    echo "No se pudo entrar a la carpeta del juego."
    read -r -p "Presiona ENTER para cerrar..."
    exit 1
}

if [ ! -d .venv ]; then
    python3 -m venv .venv || {
        echo "No se pudo crear el entorno virtual."
        read -r -p "Presiona ENTER para cerrar..."
        exit 1
    }
fi

source .venv/bin/activate || {
    echo "No se pudo activar .venv."
    read -r -p "Presiona ENTER para cerrar..."
    exit 1
}

if ! python -c "from importlib.metadata import version; from packaging.requirements import Requirement; specs=('pygame>=2.6.1','numpy>=1.26,<2.3','opencv-python-headless>=4.12,<4.13','scikit-learn>=1.5','torch>=2.4,<3','Pillow>=10','certifi>=2024.8.30'); assert all(Requirement(spec).specifier.contains(version(Requirement(spec).name)) for spec in specs)" >/dev/null 2>&1; then
    python -m pip install -r requirements.txt || {
        echo "Falló la instalación de dependencias."
        read -r -p "Presiona ENTER para cerrar..."
        exit 1
    }
fi

python main.py "$@"
status=$?
if [ "$status" -ne 0 ]; then
    echo "El juego terminó con error $status. Revisa logs/error.log."
    read -r -p "Presiona ENTER para cerrar..."
fi
exit "$status"
