import os
import subprocess
import sys
from pathlib import Path


def test_headless_smoke():
    env = os.environ.copy()
    env['SDL_VIDEODRIVER'] = 'dummy'
    result = subprocess.run(
        [sys.executable, 'main.py', '--smoke'],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert 'Ventana pygame abierta' in result.stdout, result.stdout
    assert 'MODO_1_RESULTADOS: OK' in result.stdout, result.stdout
    assert 'MODO_2_RESULTADOS: OK' in result.stdout, result.stdout
    assert 'MODO_3_RESULTADOS: OK' in result.stdout, result.stdout
    assert 'SMOKE_OK' in result.stdout, result.stdout
