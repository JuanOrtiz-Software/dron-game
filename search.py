from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

Position = Tuple[int, int]

NEIGHBOR_ORDER: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class _ExpansionComplete:
    pass


def neighbors(node: Position, walkable: Optional[Set[Position]] = None) -> Iterable[Position]:
    x, y = node
    for dx, dy in NEIGHBOR_ORDER:
        candidate = (x + dx, y + dy)
        if walkable is None or candidate in walkable:
            yield candidate


class LiveSearch:
    """Generadores incrementales que conservan el orden de los BFS/DFS originales."""

    def __init__(
        self,
        start: Position,
        goal: Optional[Position],
        adjacency: Dict[Position, List[Position]],
        mode: str,
    ):
        self.start = start
        self.goal = goal
        self.adjacency = adjacency
        self.mode = mode.lower()
        self.history: List[Position] = []
        self.parent: Dict[Position, Optional[Position]] = {start: None}
        self.nodes_expanded = 0
        self.finished = False
        self.result = False
        self._visited_set: Set[Position] = set()
        self._frontier: deque[Position] | List[Position]
        if self.mode == "bfs":
            self._frontier = deque([start])
            self._visited_set.add(start)
            self._generator = self._bfs_generator()
        elif self.mode == "dfs":
            self._frontier = [start]
            self._generator = self._dfs_generator()
        else:
            raise ValueError(f"Modo no soportado: {mode}")

    @property
    def frontier(self) -> Tuple[Position, ...]:
        return tuple(self._frontier)

    @property
    def visited_set(self) -> frozenset[Position]:
        return frozenset(self._visited_set)

    @property
    def explored(self) -> frozenset[Position]:
        return self.visited_set

    def _bfs_generator(self) -> Iterator[Position | _ExpansionComplete]:
        visitados = self.history
        visitados_set = self._visited_set
        cola = self._frontier
        assert isinstance(cola, deque)

        while cola:
            nodo = cola.popleft()
            visitados.append(nodo)
            yield nodo

            if nodo == self.goal:
                return True

            for vecino in self.adjacency.get(nodo, []):
                if vecino not in visitados_set:
                    visitados_set.add(vecino)
                    self.parent[vecino] = nodo
                    cola.append(vecino)
            yield _ExpansionComplete()

        return False

    def _dfs_generator(self) -> Iterator[Position | _ExpansionComplete]:
        visitados = self.history
        visitados_set = self._visited_set
        pila = self._frontier
        assert isinstance(pila, list)

        while pila:
            nodo = pila.pop()

            if nodo in visitados_set:
                continue

            visitados.append(nodo)
            visitados_set.add(nodo)
            yield nodo

            if nodo == self.goal:
                return True

            for vecino in self.adjacency.get(nodo, []):
                if vecino not in visitados_set:
                    self.parent.setdefault(vecino, nodo)
                    pila.append(vecino)
            yield _ExpansionComplete()

        return False

    def step(self) -> Optional[Position]:
        if self.finished:
            return None
        try:
            node = next(self._generator)
        except StopIteration as stopped:
            self.finished = True
            self.result = bool(stopped.value)
            return None

        if isinstance(node, _ExpansionComplete):
            raise RuntimeError("La búsqueda dejó un límite de expansión sin consumir.")
        self.nodes_expanded += 1
        try:
            boundary = next(self._generator)
        except StopIteration as stopped:
            self.finished = True
            self.result = bool(stopped.value)
        else:
            if not isinstance(boundary, _ExpansionComplete):
                raise RuntimeError("La búsqueda no terminó una expansión antes de continuar.")
        return node

    def path_to(self, target: Position) -> List[Position]:
        if target not in self.parent:
            return []
        path: List[Position] = []
        cursor: Optional[Position] = target
        while cursor is not None:
            path.append(cursor)
            cursor = self.parent[cursor]
        path.reverse()
        return path


def transit_path(start: Position, goal: Position, walkable: Set[Position]) -> List[Position]:
    """Ruta de movimiento sobre celdas ya conocidas; no decide el orden de búsqueda."""
    if start == goal:
        return [start]
    if start not in walkable or goal not in walkable:
        raise ValueError("El tránsito requiere inicio y destino ya explorados.")

    queue: deque[Position] = deque([start])
    previous: Dict[Position, Optional[Position]] = {start: None}

    while queue and goal not in previous:
        current = queue.popleft()
        for neighbor in neighbors(current, walkable):
            if neighbor in previous:
                continue
            previous[neighbor] = current
            queue.append(neighbor)
            if neighbor == goal:
                break

    if goal not in previous:
        raise ValueError(f"No existe tránsito conocido entre {start} y {goal}.")

    path: List[Position] = []
    cursor: Optional[Position] = goal
    while cursor is not None:
        path.append(cursor)
        cursor = previous[cursor]
    path.reverse()
    return path
