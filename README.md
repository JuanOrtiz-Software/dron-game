# DRONE WARS · CIUDAD DE JUGUETE

Juego en Python para explorar una ciudad por cuadrícula y comparar dos drones con BFS/DFS. Cada dron observa las celdas cercanas y mantiene su propia niebla de guerra. El arte del juego se genera localmente si no hay imágenes descargadas o propias.

## Abrir el juego

En macOS, haz doble clic en `run_mac.command`. También puedes abrir Terminal y ejecutar:

```bash
cd "/ruta/a/dron_game"
./run_mac.command
```

En Windows abre `run_windows.bat`; en Linux ejecuta `./run_linux.sh`. Los lanzadores crean `.venv`, instalan `requirements.txt` si faltan dependencias y dejan la terminal abierta si algo falla.

Para iniciarlo manualmente:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

La ventana es de pygame y funciona con el renderizado normal de escritorio. Los sprites se crean y cargan en el hilo gráfico principal; la preparación del modelo y las descargas HTTPS opcionales trabajan en segundo plano y comunican su estado mediante una cola.

## Diagnóstico y pruebas

```bash
python main.py --selfcheck
SDL_VIDEODRIVER=dummy python main.py --smoke
SDL_VIDEODRIVER=dummy python main.py --stress
SDL_VIDEODRIVER=dummy python main.py --capture-evidence
SDL_VIDEODRIVER=dummy python benchmark.py --mode2-seeds 30 --mode3-seeds 20
```

`--selfcheck` revisa Python, paquetes, SDL, ventana, sprites, modelo, inferencia, permisos e Internet/SSL; guarda `docs/screenshots/selfcheck.png`. `--smoke` abre una ventana pygame real y dibuja cada partida hasta los resultados. `--capture-evidence` crea capturas del render real en `docs/screenshots/`.

`--stress` ejecuta 200 semillas por modo hasta resultados, además de mapas aleatorios pequeños/densos y estados límite de búsqueda, modelo y sprites. Usa un clasificador determinista vacío para concentrar esta prueba en el bucle de juego; `--selfcheck` valida la CNN real.

La comprobación interactiva en hardware está enumerada en [`docs/MANUAL_CHECKLIST.md`](docs/MANUAL_CHECKLIST.md); no se considera cumplida por las pruebas headless.

## Juego

- **BUSCAR Y DESTRUIR:** BFS (rojo) y DFS (azul) exploran simultáneamente; solo disparan a objetivos reconocidos dentro de su visión. La meta se crea al destruir los tanques o al agotarse los barridos de seguridad.
- **¿QUIÉN GANA?:** eliges qué dron juega primero. Cada algoritmo juega solo en el mismo mundo restaurado y luego se comparan sus resultados.
- **MISIÓN CON RECURSOS LIMITADOS:** visión reducida, batería y munición limitadas, tres estaciones; los drones pueden recargar y volver al punto donde interrumpieron la búsqueda.
- El mundo predeterminado es de 20 × 20, con 20 tanques, 10 vehículos, 30 edificios y orientaciones reproducibles por semilla. Los valores se configuran y validan en `config.py`.
- Las partidas guardan sus métricas por dron en `results/` como JSON. `--smoke` ejecuta y comprueba los tres modos sin usar la pantalla física.
- Usa `ESC` solo desde el menú principal o el botón **SALIR** para cerrar.
- Los resultados incluyen una comparación visual y permiten iniciar otra partida o volver al menú.

## Algoritmos

Las funciones originales `bfs` y `dfs` se conservan como oráculo en `reference/reference_algorithms.py`, junto con la construcción de adyacencias desde una cuadrícula. `search.py` las adapta a generadores incrementales: BFS mantiene `deque`/`popleft()` y marca al encolar; DFS mantiene una pila/lista y marca al extraer. Se conserva el orden original de vecinos y la condición de meta; cada generador entrega los nodos en el mismo orden del recorrido de referencia.

La única lógica auxiliar de rutas es `transit_path`: mueve el dron por celdas ya exploradas entre el dron y el siguiente nodo entregado, sin escoger qué nodo expande BFS/DFS. Los viajes a estaciones usan el mismo generador y reconstruyen el camino con sus padres.

`tests/test_search_equivalence.py` compara ambas versiones en 300 cuadrículas aleatorias (600 comparaciones de algoritmo), y comprueba el momento de marcado, el orden de exploración de ambos drones y que el tránsito no sustituya la búsqueda. `--selfcheck` repite una comprobación corta de integridad.

### Mapa de pruebas

| Archivo / comando | Cobertura principal |
|---|---|
| `tests/test_search_equivalence.py` | Oráculos BFS/DFS, 300 mapas aleatorios, estados límite de búsqueda, tránsito, sprites y regresión del crash del turno final. |
| `tests/test_p0_gameplay.py` | Generación/restauración del mundo, visión CNN, pipeline de fotos, licencias, recursos y estadísticas. |
| `tests/test_smoke.py` | Ciclo headless de los tres modos y renderizado de resultados. |
| `python main.py --stress` | 200 semillas por modo y casos límite aleatorios/instrumentados, incluido modelo y sprite ausentes. |

Ejecuta la suite con `.venv/bin/python -m pytest -q`. Las pruebas de pygame usan SDL dummy; la interacción física sigue pendiente según [`docs/MANUAL_CHECKLIST.md`](docs/MANUAL_CHECKLIST.md).

## Datos y activos

- `models/vision.pt`: CNN PyTorch ejecutada en CPU.
- `models/vision_sklearn.joblib`: fallback HOG + SVM.
- `models/metrics.json`: exactitud y matriz de confusión.
- `models/model_info.json`: backend activo, fuentes y configuración de evaluación.
- `assets/MANIFEST.md`: listado y origen de los sprites.
- `assets/CREDITS.md`: atribución de los activos.
- `assets/user_tanks/`: imágenes propias de tanques. Añade 8–15 fotos de juguetes distintos, con ángulos, iluminación y fondos variados; se procesan con recorte, RGBA y sprite destruido, aparecen en el juego y entran al entrenamiento. El proyecto no inventa fotos si esta carpeta está vacía.

El modelo se entrena con sprites que también utiliza el juego y aumentos de imagen de OpenCV. Las fuentes de validación se mantienen fuera del entrenamiento y las métricas y la matriz se guardan en `models/metrics.json`. El resultado con arte sintético no sustituye una validación con fotografías reales. Usa **REENTRENAR MODELO** desde el menú después de añadir imágenes.

La descarga opcional consulta la API oficial de Wikimedia Commons, exige HTTPS con certificados de `certifi` y conserva atribución/licencia. Solo acepta CC0, dominio público o CC BY (no CC BY-SA); valida y procesa la imagen. Si no encuentra una licencia admitida o no hay Internet, el juego registra el motivo y conserva el arte procedural. Nunca se desactiva la verificación SSL.

Para entrenar manualmente:

```bash
.venv/bin/python train_vision.py --epochs 80
```

Para repetir la comparación algorítmica y el balance de recursos con el backend visual activo:

```bash
.venv/bin/python benchmark.py --mode2-seeds 30 --mode3-seeds 20
```

El benchmark guarda detalle y promedios en `results/benchmark.json` y tablas CSV por modo. No usa el clasificador artificial del smoke.

### Balance medido (CNN, CPU)

Promedios de 30 semillas del modo 2 y 20 del modo 3, semillas 100 en adelante. Los tiempos son segundos simulados, no tiempo de pared.

| Modo | Algoritmo | Victorias | Score | Tiempo (s) | Pasos reales | Nodos expandidos | Recargas | FP / FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | BFS | — | 20,0 | 660,0 | 2 478,0 | 495,6 | — | 0 / 0 |
| 2 | DFS | — | 20,0 | 178,5 | 551,8 | 465,0 | — | 0 / 0 |
| 3 | BFS | 0 % | 4,4 | 607,8 | 1 510,3 | 1 152,4 | 5,2 | 0 / 0 |
| 3 | DFS | 100 % | 15,6 | 389,7 | 700,5 | 1 217,0 | 2,2 | 0 / 0 |

El benchmark no respalda que el modo 3 esté equilibrado: DFS ganó las 20 semillas y destruyó más objetivos en promedio. Pruebas exploratorias de cinco semillas con batería simétrica mayor, recargas más rápidas y radio de visión 2 mantuvieron 0/5 victorias de BFS, así que los valores de producción no se cambiaron. Las funciones BFS/DFS no se alteraron ni se aplicó una ventaja asimétrica; lograr un 40–60 % por algoritmo puede requerir cambiar la regla de competencia, no solo el balance de recursos.

## SI ALGO FALLA

1. **pygame no se instala:** actualiza las herramientas y vuelve a instalar desde el entorno virtual:
   ```bash
   python3 -m pip install --upgrade pip setuptools wheel
   python3 -m pip install "pygame>=2.6.1"
   ```
2. **PyTorch no se instala:** la CNN principal requiere PyTorch; el modo alternativo HOG/SVM necesita OpenCV y scikit-learn. Reinstala los paquetes compatibles con tu sistema:
   ```bash
   python3 -m pip install -r requirements.txt
   ```
3. **Falla SSL al descargar activos:** actualiza los certificados. La descarga es opcional; al desconectarse de Internet se usan los sprites locales:
   ```bash
   python3 -m pip install --upgrade certifi
   python3 main.py --selfcheck
   ```
   Para la instalación de Python.org en macOS, también puedes abrir `Applications/Python 3.x/Install Certificates.command`.
4. **Ventana negra o en blanco:** comprueba el renderizado y vuelve a abrir con SDL por software:
   ```bash
   python3 main.py --selfcheck
   SDL_RENDER_DRIVER=software python3 main.py --debug
   ```
5. **La ventana parece cerrarse al iniciar:** usa el lanzador para mantener la Terminal abierta y obtener el traceback:
   ```bash
   ./run_mac.command
   ```
   Luego copia aquí el mensaje de Terminal y `logs/error.log`.
6. **El primer entrenamiento tarda:** déjalo terminar; se ejecuta en segundo plano sin bloquear eventos. Puedes cancelar la preparación para continuar con visión básica. Si el modelo quedó incompleto, vuelve a ejecutarlo:
   ```bash
   python3 main.py
   ```
7. **La tipografía no está disponible:** el juego intenta DejaVu Sans, Helvetica y Liberation Sans, y termina usando la fuente incluida con pygame. No requiere instalar Arial.
8. **Error de permisos al guardar:** ejecuta el proyecto desde una carpeta propia con permiso de escritura y revisa:
   ```bash
   python3 main.py --selfcheck
   chmod u+rwX logs results models assets
   ```
