
import pygame
import time
import random
from collections import deque

# --- Configuración ---
GRID_SIZE = 20
OBSTACLE_PERCENT = 0.2
CELL_SIZE = 30
DELAY = 0.1
PANEL_HEIGHT = 60

# --- Colores ---
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
LIGHT_BLUE = (180, 210, 255)
DARK_GREEN = (0, 120, 0)
DARK_RED = (180, 0, 0)

# --- Inicializar Pygame ---
pygame.init()

window = pygame.display.set_mode(
    (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE + PANEL_HEIGHT)
)
pygame.display.set_caption("BFS - Búsqueda primero en amplitud")

clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24, bold=True)

# --- Generar mundo con obstáculos ---
grid = [
    [0 for _ in range(GRID_SIZE)]
    for _ in range(GRID_SIZE)
]

for y in range(GRID_SIZE):
    for x in range(GRID_SIZE):
        if random.random() < OBSTACLE_PERCENT:
            grid[y][x] = 1

# --- Crear diccionario de conexiones ---
arbol = {}

for y in range(GRID_SIZE):
    for x in range(GRID_SIZE):

        if grid[y][x] == 0:
            vecinos = []

            for dx, dy in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1)
            ]:
                nx_, ny_ = x + dx, y + dy

                if 0 <= nx_ < GRID_SIZE and 0 <= ny_ < GRID_SIZE:
                    if grid[ny_][nx_] == 0:
                        vecinos.append((nx_, ny_))

            arbol[(x, y)] = vecinos

# --- Posición inicial ---
start = (0, 0)

while grid[start[1]][start[0]] == 1:
    start = (
        random.randint(0, GRID_SIZE - 1),
        random.randint(0, GRID_SIZE - 1)
    )

# --- Meta aleatoria ---
while True:
    goal = (
        random.randint(0, GRID_SIZE - 1),
        random.randint(0, GRID_SIZE - 1)
    )

    if grid[goal[1]][goal[0]] == 0 and goal != start:
        break

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
            print(
                f"¡META ENCONTRADA! "
                f"Se visitaron {len(visitados)} celdas."
            )
            return visitados, True

        # Agregar vecinos a la cola
        for vecino in arbol.get(nodo, []):

            if vecino not in visitados_set:
                visitados_set.add(vecino)
                cola.append(vecino)

    print("FIN: No se encontró una ruta hacia la meta.")
    return visitados, False

# --- Ejecutar BFS ---
recorrido, encontrada = bfs(start, goal, arbol)

print(f"Nodos visitados: {len(recorrido)}")

# --- Función para dibujar ---
def dibujar_mundo(agent_pos=None, explorados=None, mensaje=""):

    window.fill(WHITE)

    explorados_set = set(explorados or [])

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):

            rect = pygame.Rect(
                x * CELL_SIZE,
                y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )

            if grid[y][x] == 1:
                color = BLACK
            elif (x, y) in explorados_set:
                color = LIGHT_BLUE
            else:
                color = GRAY

            pygame.draw.rect(window, color, rect)
            pygame.draw.rect(window, WHITE, rect, 1)

    # Inicio
    sx, sy = start
    pygame.draw.rect(
        window,
        GREEN,
        (sx * CELL_SIZE, sy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    )

    # Meta
    gx, gy = goal
    pygame.draw.rect(
        window,
        RED,
        (gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    )

    # Agente
    if agent_pos is not None:
        ax, ay = agent_pos
        pygame.draw.rect(
            window,
            BLUE,
            (ax * CELL_SIZE, ay * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        )

    # Panel inferior
    panel_y = GRID_SIZE * CELL_SIZE
    pygame.draw.rect(
        window,
        WHITE,
        (0, panel_y, GRID_SIZE * CELL_SIZE, PANEL_HEIGHT)
    )

    if mensaje:
        color_texto = DARK_GREEN if encontrada else DARK_RED
        texto = font.render(mensaje, True, color_texto)
        texto_rect = texto.get_rect(
            center=(
                (GRID_SIZE * CELL_SIZE) // 2,
                panel_y + PANEL_HEIGHT // 2
            )
        )
        window.blit(texto, texto_rect)

    pygame.display.flip()

# --- Simulación: explorar hasta encontrar la meta ---
running = True
indice = 0
posicion_final = start

while running and indice < len(recorrido):

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not running:
        break

    posicion_final = recorrido[indice]

    # Mostrar únicamente las celdas exploradas hasta este momento
    dibujar_mundo(
        agent_pos=posicion_final,
        explorados=recorrido[:indice + 1]
    )

    # Si el agente llega al cuadro rojo, termina inmediatamente
    if posicion_final == goal:
        break

    clock.tick(60)
    time.sleep(DELAY)
    indice += 1

# --- Mostrar el resultado final ---
if running:

    if encontrada and posicion_final == goal:
        mensaje = "¡META ENCONTRADA!"
        print("La búsqueda terminó al llegar a la meta.")
    else:
        mensaje = "FIN: Meta no alcanzada"
        print("La búsqueda terminó sin alcanzar la meta.")

    dibujar_mundo(
        agent_pos=posicion_final,
        explorados=recorrido[:indice + 1],
        mensaje=mensaje
    )

# --- Mantener ventana abierta ---
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    clock.tick(30)

pygame.quit()
