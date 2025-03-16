import torch

ASTEROID_CODE = -2.0
ENEMY_SHIP_CODE = -1.0
RESOURCE_CODE = 1.0
PLANET_CODE = 0.5
ALLIED_SHIP_CODE = -0.5
EMPTY_CODE = 0.0

def encode(obs, ship, window_size=10):
    ship_x = ship["position_x"]
    ship_y = ship["position_y"]

    window = torch.zeros((2 * window_size + 1, 2 * window_size + 1))
    
    for ally_ship in obs["allied_ships"]:
        x = ally_ship["position_x"]
        y = ally_ship["position_y"]
        if x - ship_x < window_size and y - ship_y < window_size:
            window[x - ship_x + window_size, y - ship_y + window_size] = ALLIED_SHIP_CODE

    for enemy_ship in obs["enemy_ships"]:
        x = enemy_ship["position_x"]
        y = enemy_ship["position_y"]
        if x - ship_x < window_size and y - ship_y < window_size:
            window[x - ship_x + window_size, y - ship_y + window_size] = ENEMY_SHIP_CODE

    for planet in obs["planets_occupation"]:
        x = planet["position_x"]
        y = planet["position_y"]
        if x - ship_x < window_size and y - ship_y < window_size:
            window[x - ship_x + window_size, y - ship_y + window_size] = PLANET_CODE

    for point_on_map in obs["map"]:
        x = point_on_map["x"]
        y = point_on_map["y"]
        if x - ship_x < window_size and y - ship_y < window_size:
            if point_on_map["type"] == "asteroid":
                window[x - ship_x + window_size, y - ship_y + window_size] = ASTEROID_CODE
            elif point_on_map["type"] == "resource":
                window[x - ship_x + window_size, y - ship_y + window_size] = RESOURCE_CODE

    return window
