# Auditoría de avance y cumplimiento

Última revisión: 2026-11-06. Los resultados de entrenamiento y benchmarks se generaron en macOS/arm64 con Python 3.12.15, pygame 2.6.1 y PyTorch CPU. No se debe interpretar la métrica de sprites como validación con fotografías.

## Addendum: crash y algoritmos originales

- [x] Reproducido el `IndexError` del último traceback en `logs/error.log`: `main.py`, `_active_drones`, expresión `side = self.mode2_order[self.mode2_turn_index]` (en el traceback registrado, línea 452). Estado inválido para esa lectura: orden `["red", "blue"]` e índice `2`, ya incrementado al terminar el segundo turno.
- [x] Causa raíz confirmada por reproducción: `_draw_state()` entraba al caso `match`; `advance_match()` finalizaba el segundo turno y cambiaba el estado a `results`, pero el frame llamaba incondicionalmente a `_draw_match()`. Este alcanzaba `_active_drones()` con el índice terminal. Ahora se comprueba el estado después del tick y se dibuja resultados; no se limita ni se oculta el índice.
- [x] Pantalla de error: muestra tipo, mensaje y `archivo:línea (función)`; el traceback completo se imprime en Terminal y se registra. Regresión reproducible en `tests/test_search_equivalence.py`.
- [x] La repetición de benchmark CNN halló otro estado inválido antes de actualizar la tabla: modo 3, semilla 100, `drone.py` en `Drone.advance()` al llamar `transit_path()`. En el tick 1 032, BFS estaba en `(0, 0)`, el nodo recién entregado era `(6, 6)`, y la búsqueda de meta recién reiniciada solo conocía `(6, 6)`. La meta apareció durante un viaje a estación; al regresar se restauraba el tránsito de exploración suspendido junto con la búsqueda nueva de meta. Ahora `_begin_goal_phase()` descarta ese tránsito obsoleto; la regresión valida la transición y la semilla 100 termina en 3 164 ticks.
- [x] `reference/reference_algorithms.py` conserva BFS/DFS y la construcción de adyacencias del usuario; `search.py` mantiene generadores incrementales con `deque` BFS, lista DFS y el orden original. `transit_path` solo calcula tránsito físico sobre celdas conocidas.
- [x] Integración: `_create_drone()` asigna BFS al rojo/Drone 1 y DFS al azul/Drone 2; `Drone.__init__`, `_begin_sweep()`, `_begin_goal_phase()`, `_begin_resource_trip()` y `_advance_resource_trip()` crean búsquedas incrementales para exploración, meta y viajes de estación.
- [x] Equivalencia en 300 mapas (600 comparaciones BFS/DFS), incluidos meta igual al inicio e inalcanzable. Suite completa: 38 pruebas aprobadas.
- [x] Stress final posterior a ambas correcciones: 600/600 partidas hasta resultados (200 por modo), 130 casos límite y 0 excepciones no controladas. Incluye mundo mínimo denso 9×9 con 40 edificios completado en 1 483 ticks. Antes, 204 partidas fueron marcadas erróneamente por el tope de 10 000 ticks; la semilla 20461106 terminó en 16 363, así que se elevó el tope a 30 000. Una corrida intermedia encontró y corrigió un error de desempaquetado del propio harness.
- [x] Benchmarks CNN vigentes regenerados tras fijar los generadores: modo 2 (30 semillas) y modo 3 (20 semillas); no usar las métricas históricas previas.

## Resumen ejecutivo

- El juego integra tres modos, drones BFS/DFS, niebla de guerra, clasificación de celdas mediante imágenes renderizadas, combate, recursos limitados, resultados y pantalla recuperable de error.
- La CNN PyTorch está entrenada y es el backend principal; el checkpoint está en `models/vision.pt` y el fallback sklearn en `models/vision_sklearn.joblib`.
- El conjunto final tiene 800 muestras por clase de entrenamiento y 240 por clase de validación. La exactitud observada es 97.08% en aumentaciones de sprites fuente separados; no estima el desempeño sobre fotos reales.
- Hay 0 fotos del usuario y 1 foto CC0 descargada y verificada. La validación fotográfica queda pendiente hasta que se incorporen imágenes propias.
- Benchmark vigente con CNN en CPU: 30 semillas del modo 2 y 20 del modo 3; los resultados actuales están en `results/benchmark.json` y se resumen en README.
- La revisión visual cubrió capturas de carga, modelo, menú, modos 1–3, combate, meta, resultados, error, ocho cuadros de explosión y seis de movimiento. Las capturas son de SDL dummy, no certifican resolución Retina, rendimiento ni interacción física de ratón.

## Estado por requisito

### Juego y algoritmos

- [x] Generación determinista por semilla, conectividad y cantidades mínimas de objetos. Verificación: `tests/test_p0_gameplay.py`, `--selfcheck` y `--smoke`.
- [x] BFS rojo y DFS azul incrementales en el orden de los originales, con tránsito físico adyacente y resolución de turnos. Verificación: prueba de equivalencia en 300 mapas aleatorios.
- [x] Visión Chebyshev, niebla de guerra y observación mediante píxeles de celdas renderizadas. El estado real del mapa no se entrega al clasificador para tomar la decisión.
- [x] Disparos, precisión, falsos positivos/negativos, sprites destruidos, láser y explosiones. Capturas revisadas y pruebas de gameplay.
- [x] Re-barridos, aparición de objetivo, fase de meta, turnos y cronómetros por ticks. Verificación en pruebas y smoke.
- [x] Tres modos; modo 2 restaura el mundo entre partidas; modo 3 añade batería, munición, estaciones y choques. Verificación en pruebas y smoke.
- [x] HUD, comparación de resultados, gráfico de puntuación, veredicto y controles comprobados visualmente en las capturas incluidas en `docs/screenshots/`.
- [ ] Cumplimiento exacto de cada detalle visual/especificación original: parcial. Las capturas permiten revisar el aspecto general, pero no certifican cada criterio visual contra una lista exhaustiva. La arquitectura sigue concentrada en parte en `main.py`.

### Visión por computadora y datos

- [x] CNN PyTorch como backend principal y HOG/SVM como fallback; estado/backend y matriz de confusión visibles. Artefactos: `models/vision.pt`, `models/vision_sklearn.joblib`, `models/model_info.json`, `models/metrics.json`.
- [x] Entrenamiento balanceado: 800 ejemplos/clase; validación: 240 ejemplos/clase; aumentaciones separadas por sprite fuente. Exactitud: 97.08% (699/720). Tanques: 222/240; vehículos: 237/240; vacío: 240/240.
- [ ] Generalización a fotografías reales: pendiente. La validación usa aumentaciones de fuentes excluidas del entrenamiento, pero estas fuentes siguen siendo un catálogo pequeño de sprites procedurales y una sola imagen descargada.
- [ ] Entrenamiento con imágenes del usuario: pendiente; `assets/user_tanks/` solo contiene `.gitkeep` (0 imágenes).
- [x] La inferencia recibe un `numpy.ndarray` producido por el renderizado de una celda; prueba: `test_cell_image_is_composed_from_rendered_sprite_and_classifier_accepts_only_pixels`.

### Activos, interfaz y errores

- [x] Downloader consulta la API oficial de Wikimedia Commons, filtra licencias permitidas, verifica la descarga, reintenta y genera atribución. Activo obtenido: una imagen CC0; detalle en `assets/MANIFEST.md` y `assets/CREDITS.md`.
- [x] Recuento del manifiesto: 0 activos de usuario, 1 descargado y 85 procedurales. Hay 25 sprites fuente de clasificación (5 tanque, 12 vehículo y 8 vacío).
- [x] `setup_assets.py` se eliminó; la preparación de activos/modelo está integrada en el flujo actual.
- [x] Capturas generadas y revisadas: carga/modelo, menú, selección, tres modos, combate/meta, resultados/error, animación de explosión (8 cuadros) y movimiento (6 cuadros). Capturas en `docs/screenshots/`.
- [x] La pantalla de error mantiene opciones de reintento, menú y salida; el registro se identifica con ruta relativa `logs/error.log` para evitar mostrar una ruta local absoluta.
- [ ] Pruebas en ventana real de macOS/Windows/Linux, Retina, FPS objetivo y ratón físico: no ejecutadas; la validación visual se hizo en SDL dummy.

### Resultados, ingeniería y validación

- [x] Benchmarks CNN repetidos tras alinear los generadores: modo 2, 30 semillas; modo 3, 20 semillas. Detalles por partida y promedios en `results/benchmark.json` y CSV.
- [x] Tabla actualizada en README: modo 2 BFS/DFS score 20/20 y 0 FP/FN; modo 3 BFS ganó 0/20 (score 4,4) y DFS 20/20 (score 15,6), sin choques.
- [ ] Balance de modo 3 al objetivo 40–60 %: no alcanzado. En cinco semillas, batería simétrica 600/800/1 200, recarga rápida y radio 2 mantuvieron BFS en 0/5 victorias. No se modificaron reglas competitivas ni se dio una ventaja asimétrica; para avanzar hace falta autorizar un rediseño de competencia o un handicap explícito.
- [ ] Modularización completa y eliminación de todo estado mutable global: parcial; `main.py` aún conserva lógica de interfaz/orquestación y otros módulos no separan todas las responsabilidades previstas.
- [ ] Cobertura exhaustiva de cada requisito: parcial. La suite existente cubre gameplay, visión, activos y uso de pygame en workers, pero no equivale a certificar todos los escenarios de plataforma real.
- [x] Lanzadores de macOS/Linux/Windows, `--selfcheck` y `--smoke` disponibles.
- [x] Después de la corrección de tránsito obsoleto: `pytest -q` (38 passed), `--selfcheck` (`TODO OK` e integridad de algoritmos), `--smoke` (tres modos), `--capture-evidence` (`CAPTURAS_OK`), stress (600 partidas, 130 límites, 0 excepciones), `compileall` y `bash -n`.
- [x] Tabla de benchmarks CNN actualizada con 30 semillas modo 2 y 20 modo 3.

## Evidencia registrada

- `models/model_info.json`: CNN CPU, 80 épocas, 800/240 muestras por clase, 0 imágenes del usuario.
- `models/metrics.json`: exactitud 0.970833..., matriz de confusión y división por sprites fuente.
- `assets/MANIFEST.md` y `assets/CREDITS.md`: fuentes y licencias.
- `results/benchmark.json` y CSV: corridas por semilla y agregados BFS/DFS.
- `tests/test_p0_gameplay.py`: pruebas automatizadas.
- `docs/screenshots/`: evidencia visual descrita en la sección anterior.

## Límites que deben mantenerse explícitos

1. La exactitud del modelo no es una tasa de reconocimiento en fotos; se necesitan fotografías variadas propias y un conjunto de prueba independiente para afirmar eso.
2. Solo se consiguió una imagen descargable que cumple la licencia; no se presenta como un conjunto fotográfico suficiente.
3. DFS domina las corridas del modo 3 observadas. Los datos no prueban que DFS sea siempre mejor para toda configuración.
4. Capturas en SDL dummy no verifican tamaño físico de ventana, Retina, FPS en hardware ni eventos de ratón reales.
5. La auditoría no afirma que la refactorización arquitectónica completa ni todos los criterios originales estén terminados.
