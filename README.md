# Programming Challenge: Autonomous Robot Exploration

## 1. Background

Welcome to the autonomous robot exploration challenge!

You are tasked with developing an intelligent navigation algorithm for a robot operating in an unknown, 2D grid-based world. The world is filled with obstacles, and your robot's goal is to find one of the designated target locations.

The core challenge is that the robot operates under **partial observability**. It is equipped with a limited-range sensor and can only see a small, circular area around its current position. It must explore this unknown environment, build a map of what it has seen, and make intelligent decisions to reach a target efficiently.

## 2. Your Task

Your **only task** is to implement the `Planner` class in the file `candidate_planner.py`.

We have provided a very basic "random walk" implementation. You must replace it with a significantly more intelligent algorithm.

### Methods to Implement:

-   `__init__(self, map_shape, view_radius)`
    -   This method is called once when the planner is created.
    -   `map_shape`: A tuple `(height, width)` representing the total dimensions of the unknown world.
    -   `view_radius`: The radius of your robot's sensor.
    -   You can use this method to initialize any data structures you need, such as your internal map of the world.

-   `plan_next_step(self, current_pos, all_possible_targets)`
    -   This is the core decision-making function, called at every single step of the simulation.
    -   `current_pos`: Your robot's current `(x, y)` coordinates.
    -   `all_possible_targets`: A list of all potential target coordinates in the entire map.
    -   Based on the information you have gathered so far (in `self.known_map`), you must return the `(x, y)` coordinates of the **very next cell** you want to move to.

### Helper Method (Provided):

-   `update_knowledge(self, current_pos, local_view)`
    -   This method is called automatically by the simulator **before** `plan_next_step`.
    -   `local_view` is a 2D `numpy` array representing the area your robot currently sees.
    -   A correct implementation that updates `self.known_map` is already provided for you. You do not need to modify it, but you must understand that your `plan_next_step` logic will rely on the `self.known_map` that this method builds.

## 3. Rules & Constraints

1.  **You must only modify `candidate_planner.py`.** Do not change any other files.
2.  Your `plan_next_step` function must return a tuple of `(x, y)` coordinates for an **adjacent cell**.
    -   **Legal moves**: Up, Down, Left, Right.
    -   **Illegal moves**: Diagonal moves, staying in the same spot, or moving into a cell you know is an obstacle.
3.  The simulation ends if you make an illegal move, exceed the step limit, or exceed the time limit.

## 4. Development and Testing

We provide a full simulation environment for you to develop and test your algorithm.

-   **`main_simulator.py`**: The main entry point to run a simulation.
-   **`TEST_SUITE_A`**: A development test set defined within `main_simulator.py`. It contains 10 diverse maps to help you debug.
-   **`visualizer.py`**: A helper script that saves images of your robot's trajectory and explored map into the `vis/` directory. This is extremely useful for debugging.

To test your code, simply run:
```bash
python main_simulator.py
```
This will run your planner against all the test cases in `TEST_SUITE_A`.

## 5. Final Evaluation

-   Your submission will be judged by running it against a **hidden, more challenging test set (`TEST_SUITE_B`)**.
-   This final test set includes much larger maps and **very strict time limits**.
-   A simple, inefficient algorithm may pass the development tests in `TEST_SUITE_A`, but it will likely fail the `TEST_SUITE_B` evaluation due to timeouts. **Algorithm efficiency is key to success.**
-   Your final score will be based on the number of test cases passed in the final evaluation. "Strict" test cases with tight time limits are worth more points.

Good luck! 