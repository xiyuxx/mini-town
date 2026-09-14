"""Town world: map grid, locations, and spatial helpers."""

from dataclasses import dataclass, field
from typing import Optional

from .world_pack import DEFAULT_WORLD

GRID_W = int(DEFAULT_WORLD.data["map"]["width"])
GRID_H = int(DEFAULT_WORLD.data["map"]["height"])
_DEFAULT_TERRAIN = str(DEFAULT_WORLD.data["map"].get("default_terrain", "open"))


@dataclass
class Location:
    id: str
    name: str
    type: str
    x: int
    y: int
    width: int
    height: int
    color: str
    emoji: str
    owner: str | None = None
    access: dict = field(default_factory=lambda: {"mode": "public"})
    affordances: list[str] = field(default_factory=list)
    pass_through: bool = False

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height


LOCATIONS: list[Location] = [
    Location(
        id=str(raw["id"]), name=str(raw["name"]), type=str(raw["type"]),
        x=int(raw["x"]), y=int(raw["y"]), width=int(raw["width"]),
        height=int(raw["height"]), color=str(raw.get("color", "#888888")),
        emoji=str(raw.get("emoji", "")), owner=raw.get("owner"),
        access=dict(raw.get("access", {"mode": "public"})),
        affordances=[str(item) for item in raw.get("affordances", [])],
        pass_through=bool(raw.get("pass_through", False)),
    )
    for raw in DEFAULT_WORLD.locations
]
LOCATION_MAP: dict[str, Location] = {loc.id: loc for loc in LOCATIONS}


def location_affordances(location_id: str) -> list[str]:
    loc = LOCATION_MAP.get(location_id)
    return list(loc.affordances) if loc else []


def travel_cost(weather: dict | str) -> dict:
    cost = float(weather.get("travel_cost", 0.0)) if isinstance(weather, dict) else 0.0
    return {
        "value": max(0.0, min(1.0, cost)),
        "level": "low" if cost < 0.25 else "medium" if cost < 0.6 else "high" if cost < 0.9 else "severe",
    }


# Locations without a roof. Weather that makes travel expensive is the same
# weather that makes standing in one of these unpleasant, so the pack's own
# travel_cost doubles as the "seek shelter" signal.
OUTDOOR_LOCATION_TYPES = {"park", "road", "transit", "market"}


def is_outdoor(location_id: str) -> bool:
    loc = LOCATION_MAP.get(location_id)
    return bool(loc and loc.type in OUTDOOR_LOCATION_TYPES)


def is_shelter_weather(weather: dict) -> bool:
    return travel_cost(weather)["level"] != "low"


# ── Grid initialization ──────────────────────────────────────
def _build_grid() -> list[list[str]]:
    """Build a 20×14 grid from LOCATIONS. Default is road, buildings override."""
    grid = [[_DEFAULT_TERRAIN for _ in range(GRID_W)] for _ in range(GRID_H)]
    for loc in LOCATIONS:
        for dy in range(loc.height):
            for dx in range(loc.width):
                gx, gy = loc.x + dx, loc.y + dy
                if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                    grid[gy][gx] = loc.type
    return grid

GRID: list[list[str]] = _build_grid()


def is_walkable(gx: int, gy: int) -> bool:
    """Return True if the cell at (gx, gy) can be walked on."""
    if not (0 <= gx < GRID_W and 0 <= gy < GRID_H):
        return False
    return GRID[gy][gx] != "wall"  # void is empty space, walkable


def _is_walkable_for_route(gx: int, gy: int, agent_id: str | None,
                           start: tuple[int, int],
                           target_location_id: str | None,
                           start_location_id: str | None) -> bool:
    """Route-specific walkability.

    Private homes are valid start/end spaces for their owner, but should not
    become shortcuts for everyone else.
    """
    if not is_walkable(gx, gy):
        return False
    if (gx, gy) == start:
        return True
    loc = find_location(gx, gy)
    if loc and not loc.pass_through:
        if loc.id == start_location_id:
            return True
        if loc.id == target_location_id:
            return can_enter(agent_id or "", loc.id)
        return False
    return True


def bfs_path(start_x: int, start_y: int, end_x: int, end_y: int,
             agent_id: str | None = None,
             target_location_id: str | None = None) -> list[tuple[int, int]]:
    """BFS on the walkable grid from start to end. Returns list of (x,y) steps
    including both endpoints, or empty list if unreachable."""
    from collections import deque

    start = (start_x, start_y)
    start_location = find_location(start_x, start_y)
    start_location_id = start_location.id if start_location else None
    if not _is_walkable_for_route(start_x, start_y, agent_id, start, target_location_id, start_location_id):
        return []
    if not _is_walkable_for_route(end_x, end_y, agent_id, start, target_location_id, start_location_id):
        return []
    if start == (end_x, end_y):
        return [start]

    visited: set[tuple[int, int]] = {start}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    queue = deque([start])

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = cx + dx, cy + dy
            if not _is_walkable_for_route(nx, ny, agent_id, start, target_location_id, start_location_id):
                continue
            if (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            parent[(nx, ny)] = (cx, cy)
            if (nx, ny) == (end_x, end_y):
                # Reconstruct path
                path: list[tuple[int, int]] = [(end_x, end_y)]
                cur = (cx, cy)
                while cur != (start_x, start_y):
                    path.append(cur)
                    cur = parent[cur]
                path.append((start_x, start_y))
                path.reverse()
                # Movement paths contain future steps only. Including the start
                # cell makes the first movement tick appear to do nothing.
                return path[1:]
            queue.append((nx, ny))

    return []


def can_enter(agent_id: str, location_id: str) -> bool:
    """Evaluate the location's declarative access policy."""
    loc = LOCATION_MAP.get(location_id)
    if not loc:
        return False
    mode = str(loc.access.get("mode", "public"))
    allowed = {str(item) for item in loc.access.get("allowed_agents", []) if item}
    denied = {str(item) for item in loc.access.get("denied_agents", []) if item}
    if agent_id in denied:
        return False
    if mode == "public":
        return True
    if mode in {"owner_only", "allow_list"}:
        return agent_id in allowed
    return False


def find_location(x: int, y: int) -> Optional[Location]:
    """Return the location containing (x, y), preferring non-road."""
    candidates = [loc for loc in LOCATIONS if loc.contains(x, y)]
    destinations = [loc for loc in candidates if not loc.pass_through]
    return destinations[0] if destinations else (candidates[0] if candidates else None)


def location_center(loc_id: str) -> tuple[int, int]:
    loc = LOCATION_MAP.get(loc_id)
    return loc.center if loc else (0, 0)


def location_name(loc_id: str) -> str:
    if loc_id == "in_transit":
        return "路上"
    loc = LOCATION_MAP.get(loc_id)
    return loc.name if loc else "未知地点"


def agents_at_location(agents: list, loc_id: str) -> list:
    """Return agents sharing a concrete place.

    Travellers use ``in_transit`` as a display state, not a shared location;
    road encounters are resolved by exact grid coordinates in the engine.
    """
    if loc_id == "in_transit":
        return []
    return [a for a in agents if (hasattr(a, 'state') and a.state.current_location == loc_id) or getattr(a, 'current_location', None) == loc_id]
