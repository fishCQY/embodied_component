import numpy as np
import random

class Planner:
    """
    This is the planner class you need to implement.
    A very basic random-walk planner is provided as a starting point.
    """
    def __init__(self, map_shape: tuple[int, int], view_radius: int):
        """
        Initialize the planner.
        :param map_shape: Tuple of (height, width) for the entire map.
        :param view_radius: The radius of your robot's circular viewing area.
        """
        # known_map: -1 for unknown, 0 for free space, 1 for obstacle.
        self.known_map = np.full(map_shape, -1, dtype=int)
        self.view_radius = view_radius
        self.path_taken = [] # A list to store the history of positions.
        
        # You can add any other member variables you need here.
        
    def update_knowledge(self, current_pos: tuple[int, int], local_view: np.ndarray):
        """
        This function is called by the simulator to provide the robot with a new local view.
        A correct implementation to update the known_map is provided. You do not need to change this.
        """
        r = self.view_radius
        x, y = current_pos
        h, w = self.known_map.shape
        
        y_start, y_end = max(0, y - r), min(h, y + r + 1)
        x_start, x_end = max(0, x - r), min(w, x + r + 1)
        
        self.known_map[y_start:y_end, x_start:x_end] = local_view

    def plan_next_step(self, current_pos: tuple[int, int], all_possible_targets: list[tuple[int, int]]) -> tuple[int, int]:
        """
        This is the core function you need to implement.
        It should return the (x, y) coordinates of the next step.
        """
        # Store the current position in our path history
        if not self.path_taken or self.path_taken[-1] != current_pos:
            self.path_taken.append(current_pos)

        # --- YOUR PLANNING LOGIC STARTS HERE ---
        
        # A very naive implementation: just move to a random valid neighbor.
        # This is a bad strategy, as it doesn't explore or go to the target.
        # It also doesn't check if it's moving into a known wall.
        # You should replace this with a much smarter algorithm.
        motions = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Up, Down, Right, Left
        random.shuffle(motions)
        
        for dx, dy in motions:
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            nx, ny = next_pos
            
            # Check if the next position is within map bounds
            if 0 <= ny < self.known_map.shape[0] and 0 <= nx < self.known_map.shape[1]:
                # Check if we know the cell is not an obstacle
                # Note: A smarter planner would also avoid moving to -1 (unknown) if possible
                if self.known_map[ny, nx] != 1:
                    return next_pos # Return the first valid random move

        # If all neighbors are walls, stay put (this should not happen in a valid map)
        return current_pos
