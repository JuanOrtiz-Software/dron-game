"""Configuración y límites del juego."""

GRID_SIZE = 20
TICK_SECONDS = 0.25
VISION_RADIUS = 2
MODE_3_VISION_RADIUS = 1
CONFIDENCE_THRESHOLD = 0.60
TANK_COUNT = 20
VEHICLE_COUNT = 10
BUILDING_COUNT = 30
CHARGING_STATION_COUNT = 3
MAX_SWEEPS = 5
SHOT_TICKS = 8
BATTERY_MAX = 400
SHOT_BATTERY_COST = 2
AMMO_MAX = 25
LOW_BATTERY_PCT = 35
RECHARGE_TICKS = 5
VISION_NOISE_STD = 3.0
VISION_BRIGHTNESS_RANGE = 6.0


def validate_config() -> None:
    if GRID_SIZE < 2:
        raise ValueError("GRID_SIZE debe ser al menos 2.")
    if TANK_COUNT < 20:
        raise ValueError("TANK_COUNT debe ser al menos 20.")
    if VEHICLE_COUNT < 10:
        raise ValueError("VEHICLE_COUNT debe ser al menos 10.")
    if BUILDING_COUNT < 10:
        raise ValueError("BUILDING_COUNT debe ser al menos 10.")
    if CHARGING_STATION_COUNT != 3:
        raise ValueError("CHARGING_STATION_COUNT debe ser exactamente 3.")
    if not 0 <= CONFIDENCE_THRESHOLD <= 1:
        raise ValueError("CONFIDENCE_THRESHOLD debe estar entre 0 y 1.")
    if MAX_SWEEPS < 1 or SHOT_TICKS < 1:
        raise ValueError("MAX_SWEEPS y SHOT_TICKS deben ser positivos.")
    if VISION_NOISE_STD < 0 or VISION_BRIGHTNESS_RANGE < 0:
        raise ValueError("La intensidad de perturbación visual no puede ser negativa.")
    required_cells = TANK_COUNT + VEHICLE_COUNT + CHARGING_STATION_COUNT + 2 + 1
    if BUILDING_COUNT + required_cells > GRID_SIZE * GRID_SIZE:
        raise ValueError("No caben los obstáculos y entidades en la cuadrícula configurada.")


validate_config()
