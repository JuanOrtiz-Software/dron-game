@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv
    if errorlevel 1 goto failed
)
call .venv\Scripts\activate.bat
python -c "from importlib.metadata import version; from packaging.requirements import Requirement; specs=('pygame>=2.6.1','numpy>=1.26,<2.3','opencv-python-headless>=4.12,<4.13','scikit-learn>=1.5','torch>=2.4,<3','Pillow>=10','certifi>=2024.8.30'); assert all(Requirement(spec).specifier.contains(version(Requirement(spec).name)) for spec in specs)" >nul 2>nul
if errorlevel 1 (
    python -m pip install -r requirements.txt
    if errorlevel 1 goto failed
)
python main.py %*
if errorlevel 1 goto failed
exit /b 0

:failed
echo El juego o la instalacion fallo. Revisa logs\error.log.
pause
exit /b 1
