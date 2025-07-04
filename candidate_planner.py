import numpy as np
import random

class Planner:
    """
    This is the planner class that implements Q-learning.
    """
    def __init__(self, map_shape: tuple[int, int], view_radius: int, gamma=0.9, alpha=0.1, epsilon=0.1):
        """
        Initialize the planner.
        :param map_shape: Tuple of (height, width) for the entire map.
        :param view_radius: The radius of your robot's circular viewing area.
        :param gamma: Discount factor for future rewards.
        :param alpha: Learning rate.
        :param epsilon: Exploration rate for epsilon-greedy strategy.
        """
        # known_map: -1 for unknown, 0 for free space, 1 for obstacle.
        self.known_map = np.full(map_shape, -1, dtype=int)
        self.view_radius = view_radius
        self.path_taken = []  # A list to store the history of positions.
        
        # Q-table initialization (key: (x, y, local_view), value: Q-values for each action)
        self.Q = {}

        # Parameters for Q-learning
        self.gamma = gamma  # Discount factor
        self.alpha = alpha  # Learning rate
        self.epsilon = epsilon  # Exploration rate

    def update_knowledge(self, current_pos: tuple[int, int], local_view: np.ndarray):
        """
        This function is called by the simulator to provide the robot with a new local view.
        """
        r = self.view_radius
        x, y = current_pos
        h, w = self.known_map.shape
        
        y_start, y_end = max(0, y - r), min(h, y + r + 1)
        x_start, x_end = max(0, x - r), min(w, x + r + 1)
        
        self.known_map[y_start:y_end, x_start:x_end] = local_view

    def plan_next_step(self, current_pos: tuple[int, int], all_possible_targets: list[tuple[int, int]]) -> tuple[int, int]:
        """
        This is the core function that uses Q-learning to decide the next move.
        """
        # Store the current position in our path history
        if not self.path_taken or self.path_taken[-1] != current_pos:
            self.path_taken.append(current_pos)

        # Get the local view of the current position (this could be given as part of the state)
        local_view = self.known_map[max(0, current_pos[1] - self.view_radius) : current_pos[1] + self.view_radius + 1,
                                    max(0, current_pos[0] - self.view_radius) : current_pos[0] + self.view_radius + 1]

        # --- Q-learning logic: epsilon-greedy strategy ---
        actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, Down, Right, Left
        state = (current_pos, tuple(map(tuple, local_view)))  # Tuple of position and local view
        
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: Choose a random action
            action = random.choice(actions)
        else:
            # Exploitation: Choose the best action based on Q-values
            if state not in self.Q:
                self.Q[state] = {a: 0 for a in actions}  # Initialize Q-values for all actions
                
            # Choose the action with the highest Q-value
            action = max(self.Q[state], key=self.Q[state].get)

        # Apply the chosen action and calculate the next position
        next_pos = (current_pos[0] + action[0], current_pos[1] + action[1])

        # Ensure the next position is within bounds and not an obstacle
        if 0 <= next_pos[0] < self.known_map.shape[0] and 0 <= next_pos[1] < self.known_map.shape[1]:
            if self.known_map[next_pos[1], next_pos[0]] != 1:  # Avoid obstacles
                return next_pos

        # If the chosen move is invalid (obstacle or out of bounds), stay in place
        return current_pos

    def update_q_value(self, state, action, reward, next_state):
        """
        Update the Q-value using the Q-learning update rule.
        """
        if state not in self.Q:
            self.Q[state] = {a: 0 for a in [(0, 1), (0, -1), (1, 0), (-1, 0)]}
        
        if next_state not in self.Q:
            self.Q[next_state] = {a: 0 for a in [(0, 1), (0, -1), (1, 0), (-1, 0)]}
        
        # Q-learning update rule
        max_q_next = max(self.Q[next_state].values())
        self.Q[state][action] += self.alpha * (reward + self.gamma * max_q_next - self.Q[state][action])
