import time
import numpy as np
import os
from visualizer import visualize_exploration_step
from candidate_planner import Planner

def run_exploration_test(planner_class, test_config):
    """
    根据详细的测试配置，运行一次探索与规划测试。
    """
    case_name = os.path.basename(test_config['map_file']).replace('.npz', '')
    run_name = f"{case_name}_radius_{test_config['radius']}"
    print(f"\n--- Running Test: {run_name} (Step Limit: {test_config['step_limit']}, Time Limit: {test_config['time_limit']}s) ---")

    data = np.load(test_config['map_file'])
    true_map, start_pos, targets = data['grid_map'], tuple(data['start_pos']), {tuple(row) for row in data['target_positions']}

    planner = planner_class(map_shape=true_map.shape, view_radius=test_config['radius'])
    
    current_pos = start_pos
    total_steps = 0
    start_time = time.time()

    while total_steps < test_config['step_limit']:
        duration = time.time() - start_time
        if duration > test_config['time_limit']:
            print(f"Result: FAILED (Timeout: {duration:.2f}s)")
            visualize_exploration_step(f"FAIL_timeout: {run_name}", f"{run_name}_fail_timeout", true_map, planner.known_map, planner.path_taken, current_pos, targets)
            return "Timeout", total_steps

        r = test_config['radius']
        x, y = current_pos
        h, w = true_map.shape
        local_view = true_map[max(0, y-r):min(h, y+r+1), max(0, x-r):min(w, x+r+1)]

        planner.update_knowledge(current_pos, local_view)
        next_pos = planner.plan_next_step(current_pos, list(targets))
        
        if not (isinstance(next_pos, tuple) and len(next_pos) == 2 and 
                isinstance(next_pos[0], (int, np.integer)) and 
                isinstance(next_pos[1], (int, np.integer))):
            print("Result: FAILED (Invalid return type from planner)")
            visualize_exploration_step(f"FAIL_return: {run_name}", f"{run_name}_fail_return", true_map, planner.known_map, planner.path_taken, current_pos, targets)
            return "Invalid Return", total_steps

        manhattan_dist = abs(next_pos[0] - current_pos[0]) + abs(next_pos[1] - current_pos[1])
        if manhattan_dist > 1:
            print("Result: FAILED (Illegal move - not adjacent)")
            visualize_exploration_step(f"FAIL_move: {run_name}", f"{run_name}_fail_illegal_move", true_map, planner.known_map, planner.path_taken, current_pos, targets)
            return "Illegal Move", total_steps
        if true_map[next_pos[1], next_pos[0]] == 1:
            print("Result: FAILED (Collision with wall)")
            visualize_exploration_step(f"FAIL_collision: {run_name}", f"{run_name}_fail_collision", true_map, planner.known_map, planner.path_taken, current_pos, targets)
            return "Collision", total_steps

        current_pos = next_pos
        total_steps += 1
        
        if current_pos in targets:
            duration = time.time() - start_time
            print(f"Result: SUCCESS! (Total steps: {total_steps}, Time: {duration:.2f}s)")
            planner.path_taken.append(current_pos)
            visualize_exploration_step(f"SUCCESS: {run_name}", f"{run_name}_success", true_map, planner.known_map, planner.path_taken, current_pos, targets)
            return "Success", total_steps

    print(f"Result: FAILED (Exceeded step limit)")
    visualize_exploration_step(f"FAIL_steps: {run_name}", f"{run_name}_fail_step_limit", true_map, planner.known_map, planner.path_taken, current_pos, targets)
    return "Step Limit", total_steps

if __name__ == "__main__":
    # This is the development test suite for contestants to use.
    TEST_SUITE_A = [
        # Simple cases
        {"map_file": "test_maps_hard/set_A_for_candidates/case_standard_1.npz", "radius": 10, "step_limit": 500, "time_limit": 20},
        {"map_file": "test_maps_hard/set_A_for_candidates/case_standard_2.npz", "radius": 5, "step_limit": 2000, "time_limit": 30},
        
        # Mazes
        {"map_file": "test_maps_hard/set_A_for_candidates/case_maze_1.npz", "radius": 10, "step_limit": 1500, "time_limit": 20},
        {"map_file": "test_maps_hard/set_A_for_candidates/case_maze_2.npz", "radius": 1000, "step_limit": 3000, "time_limit": 20},
        {"map_file": "test_maps_hard/set_A_for_candidates/case_maze_3.npz", "radius": 5, "step_limit": 8000, "time_limit": 40},
        {"map_file": "test_maps_hard/set_A_for_candidates/case_maze_4.npz", "radius": 20, "step_limit": 6000, "time_limit": 40},   
    ]
    # The final evaluation will use a different, hidden test suite (TEST_SUITE_B).
    
    active_suite = TEST_SUITE_A

    all_results = []
    for test_config in active_suite:
        if not os.path.exists(test_config['map_file']):
            print(f"WARNING: Map file not found, skipping test -> {test_config['map_file']}")
            continue
        
        status, steps = run_exploration_test(Planner, test_config)
        all_results.append({**test_config, "status": status, "steps": steps})

    # --- Print final summary report ---
    print("\n" + "="*80)
    print("FINAL SUMMARY REPORT (Set A):")
    print("="*80)
    header = f"{'Map File':<45} | {'Radius':<6} | {'Status':<15} | Steps Taken"
    print(header)
    print("-" * len(header))
    for r in all_results:
        map_name = os.path.basename(r['map_file'])
        radius_str = "global" if r['radius'] >= 1000 else str(r['radius'])
        status_color = "\033[92m" if r['status'] == 'Success' else "\033[91m"
        print(f"{map_name:<45} | {radius_str:<6} | {status_color}{r['status']:<15}\033[0m | {r['steps']}")
