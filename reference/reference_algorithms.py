# Copias originales de BFS/DFS del usuario, conservadas como oráculo de referencia.

from collections import deque


def arbol_desde_grid(grid):
    arbol = {}
    GRID_SIZE = len(grid)

    # --- Crear diccionario de conexiones ---
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid[y][x] == 0:
                vecinos = []

                for dx, dy in [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ]:
                    nx_, ny_ = x + dx, y + dy

                    if 0 <= nx_ < GRID_SIZE and 0 <= ny_ < GRID_SIZE:
                        if grid[ny_][nx_] == 0:
                            vecinos.append((nx_, ny_))

                arbol[(x, y)] = vecinos

    return arbol


# --- BFS: búsqueda primero en amplitud ---
def bfs(inicio, meta, arbol):

    visitados = []
    visitados_set = set()
    cola = deque([inicio])

    visitados_set.add(inicio)

    while cola:

        nodo = cola.popleft()
        visitados.append(nodo)

        # Si el agente llega a la meta, termina la búsqueda
        if nodo == meta:
            return visitados, True

        # Agregar vecinos a la cola
        for vecino in arbol.get(nodo, []):

            if vecino not in visitados_set:
                visitados_set.add(vecino)
                cola.append(vecino)

    return visitados, False


# --- DFS: búsqueda en profundidad ---
def dfs(inicio, meta, arbol):
    visitados = []
    visitados_set = set()

    # Pila: último en entrar, primero en salir
    pila = [inicio]

    while pila:
        nodo = pila.pop()

        if nodo in visitados_set:
            continue

        visitados.append(nodo)
        visitados_set.add(nodo)

        # Detener la búsqueda al encontrar la meta
        if nodo == meta:
            return visitados, True

        # Agregar vecinos a la pila
        for vecino in arbol.get(nodo, []):
            if vecino not in visitados_set:
                pila.append(vecino)

    return visitados, False
