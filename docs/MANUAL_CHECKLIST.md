# Lista de comprobación manual

Las pruebas headless no reemplazan esta revisión en una ventana de escritorio. Marca cada casilla solo después de comprobarla en el sistema indicado.

## Inicio y modelo

- [ ] Abrir con `run_mac.command` y confirmar que la ventana responde y se puede redimensionar.
- [ ] Confirmar en el resumen que el backend de visión está disponible y que muestra sus métricas.
- [ ] Cerrar y volver a iniciar; confirmar que carga sprites y modelo sin bloquear la interfaz.
- [ ] Probar «REENTRENAR MODELO» y confirmar que el progreso y los errores se muestran en español.

## Partida

- [ ] Modo 1: comprobar niebla individual, desplazamiento celda a celda, clasificación visible, disparo, destrucción y animación.
- [ ] Modo 1: verificar re-barridos, aparición de la meta y llegada de ambos drones sin saltos.
- [ ] Modo 2: alternar quién juega primero; confirmar la restauración idéntica del mundo y dos estadísticas independientes.
- [ ] Modo 3: observar gasto/recarga de batería y munición, parada en estación y retorno al punto suspendido.
- [ ] En los tres modos: dejar terminar la partida y comprobar que aparecen comparación, gráfico y veredicto; usar «MENÚ» y «JUGAR DE NUEVO».

## Recuperación y límites

- [ ] Provocar un error controlado en una compilación de prueba; confirmar tipo de excepción, mensaje, `archivo:línea`, traceback en Terminal y registro en `logs/error.log`.
- [ ] Probar el botón «REINTENTAR», el retorno al menú, `ESC` y el cierre de ventana.
- [ ] Repetir en el tamaño de pantalla habitual y, si aplica, en una pantalla Retina; observar recortes, legibilidad y respuesta del ratón.
- [ ] Si se usan fotos propias, añadir una muestra pequeña a `assets/user_tanks/`, reentrenar y revisar manifiesto, créditos y aspecto de los sprites.

## Resultado de esta ejecución

- Headless: `pytest`, `--selfcheck`, `--smoke` y `--stress` se registran en `PROGRESS.md`.
- Interacción física, resolución de escritorio y fotos propias: pendiente de ejecutar por una persona en el equipo correspondiente.
