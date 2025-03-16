# Skeleton for Agent class

alternator = False

id_dict = {}


def get_id(id):
    global id_dict

    if id in id_dict:
        return id_dict[id]
    id_dict[id] = len(id_dict)
    return id_dict[id]


def get_walk_action(id, x, y, target_x, target_y, policy):
    if policy == "diagonal":
        if alternator:
            policy = "x"
        else:
            policy = "y"
    if policy == "x":
        if x < target_x:
            return [id, 0, 0, 3]
        elif x > target_x:
            return [id, 0, 2, 3]
        elif y < target_y:
            return [id, 0, 1, 3]
        elif y > target_y:
            return [id, 0, 3, 3]
    elif policy == "y":
        if y < target_y:
            return [id, 0, 1, 3]
        elif y > target_y:
            return [id, 0, 3, 3]
        elif x < target_x:
            return [id, 0, 0, 3]
        elif x > target_x:
            return [id, 0, 2, 3]
    return [id, 0, 0, 0]


def find_enemies(id, x, y, enemies):
    for enemy in enemies:
        enemy_id, enemy_x, enemy_y, enemy_hp, enemy_fire_cd, enemy_move_cd = enemy
        if abs(enemy_x - x) < 9 and enemy_y == y:
            if enemy_x < x:
                return [id, 1, 2]
            else:
                return [id, 1, 0]
        if abs(enemy_y - y) < 9 and enemy_x == x:
            if enemy_y < y:
                return [id, 1, 3]
            else:
                return [id, 1, 1]
    return None


class Agent:
    def __init__(self, side=0) -> None:
        self.side = side % 2

        global id_dict
        id_dict = {}

    def get_action(self, obs: dict) -> dict:
        global alternator
        """
        Main function, which gets called during step() of the environment.

        Observation space:
            game_map: whole grid of board_size, which already has applied visibility mask on it
            allied_ships: an array of all currently available ships for the player. The ships are represented as a list:
                (ship id, position x, y, current health points, firing_cooldown, move_cooldown)
                - ship id: int [0, 1000]
                - position x: int [0, 100]
                - position y: int [0, 100]
                - health points: int [1, 100]
                - firing_cooldown: int [0, 10]
                - move_cooldown: int [0, 3]
            enemy_ships: same, but for the opposing player ships
            planets_occupation: for each visible planet, it shows the occupation progress:
                - planet_x: int [0, 100]
                - planet_y: int [0, 100]
                - occupation_progress: int [-1, 100]:
                    -1: planet is unoccupied
                    0: planet occupied by the 1st player
                    100: planet occupied by the 2nd player
                    Values between indicate an ongoing conflict for the ownership of the planet
            resources: current resources available for building

        Action space:
            ships_actions: player can provide an action to be executed by every of his ships.
                The command looks as follows:
                - ship_id: int [0, 1000]
                - action_type: int [0, 1]
                    0 - move
                    1 - fire
                - direction: int [0, 3] - direction of movement or firing
                    0 - right
                    1 - down
                    2 - left
                    3 - up
                - speed (not applicable when firing): int [0, 3] - a number of fields to move
            construction: int [0, 10] - a number of ships to be constructed

        :param obs:
        :return:
        """
        ship_actions = []

        for ship in obs["allied_ships"]:
            ship_id, x, y, hp, fire_cd, move_cd = ship
            strat_id = get_id(ship_id)

            enemies_found = find_enemies(ship_id, x, y, obs["enemy_ships"])
            if enemies_found is not None:
                ship_actions.append(enemies_found)
                continue

            if strat_id < 4:
                if self.side == 0:
                    ship_actions.append(get_walk_action(ship_id, x, y, 15, 15, "x"))
                if self.side == 1:
                    ship_actions.append(get_walk_action(ship_id, x, y, 85, 85, "x"))
            elif strat_id == 4:
                if self.side == 0:
                    ship_actions.append(get_walk_action(ship_id, x, y, 10, 10, "x"))
                if self.side == 1:
                    ship_actions.append(get_walk_action(ship_id, x, y, 90, 90, "x"))
            else:
                if self.side == 0:
                    ship_actions.append(get_walk_action(ship_id, x, y, 90, 90, "x"))
                else:
                    ship_actions.append(get_walk_action(ship_id, x, y, 10, 10, "x"))

        alternator = not alternator

        return {
            "ships_actions": ship_actions,
            "construction": 10
        }

    def load(self, abs_path: str):
        """
        Function for loading all necessary weights for the agent. The abs_path is a path pointing to the directory,
        where the weights for the agent are stored, so remember to join it to any path while loading.

        :param abs_path:
        :return:
        """
        pass

    def eval(self):
        """
        With this function you should switch the agent to inference mode.

        :return:
        """
        pass

    def to(self, device):
        """
        This function allows you to move the agent to a GPU. Please keep that in mind,
        because it can significantly speed up the computations and let you meet the time requirements.

        :param device:
        :return:
        """
        pass
