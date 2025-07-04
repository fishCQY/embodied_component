import time
import numpy as np
import os
from visualizer import visualize_exploration_step
from candidate_planner import Planner

def run_dynamic_visualization(planner_class, test_config, save_gif=False, vis_dir='dynamic_vis'):
    """
    运行探索并动态保存每一步的可视化图片，可选合成GIF。
    图片和GIF保存到 vis_dir 文件夹。
    """
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    case_name = os.path.basename(test_config['map_file']).replace('.npz', '')
    run_name = f"{case_name}_radius_{test_config['radius']}_dynamic"
    print(f"\n--- Dynamic Visualization: {run_name} ---")

    data = np.load(test_config['map_file'])
    true_map, start_pos, targets = data['grid_map'], tuple(data['start_pos']), {tuple(row) for row in data['target_positions']}

    planner = planner_class(map_shape=true_map.shape, view_radius=test_config['radius'])
    current_pos = start_pos
    total_steps = 0
    step_imgs = []

    while total_steps < test_config['step_limit']:
        r = test_config['radius']
        x, y = current_pos
        h, w = true_map.shape
        local_view = true_map[max(0, y-r):min(h, y+r+1), max(0, x-r):min(w, x+r+1)]
        planner.update_knowledge(current_pos, local_view)
        next_pos = planner.plan_next_step(current_pos, list(targets))

        # 保存每一步的可视化图片到 vis_dir
        img_name = f"{run_name}_step_{total_steps:04d}"
        img_path = os.path.join(vis_dir, f"{img_name}.png")
        visualize_exploration_step(f"Step {total_steps}", img_name, true_map, planner.known_map, planner.path_taken, current_pos, targets)
        # 移动图片到目标目录（visualizer默认存到vis/）
        default_img_path = f"vis/{img_name}.png"
        if os.path.exists(default_img_path):
            os.replace(default_img_path, img_path)
        step_imgs.append(img_path)

        if current_pos in targets:
            print(f"Success at step {total_steps}")
            break

        if not (isinstance(next_pos, tuple) and len(next_pos) == 2 and \
                isinstance(next_pos[0], (int, np.integer)) and \
                isinstance(next_pos[1], (int, np.integer))):
            print("Invalid return, abort.")
            break

        if abs(next_pos[0] - current_pos[0]) + abs(next_pos[1] - current_pos[1]) > 1:
            print("Illegal move, abort.")
            break

        if true_map[next_pos[1], next_pos[0]] == 1:
            print("Collision, abort.")
            break

        current_pos = next_pos
        total_steps += 1

    # 可选：合成GIF
    if save_gif and step_imgs:
        try:
            import imageio
            # 确保所有图片为uint8格式的ndarray
            images = [np.asarray(imageio.v2.imread(img)) for img in step_imgs]
            gif_path = os.path.join(vis_dir, f'{run_name}.gif')
            imageio.mimsave(gif_path, list(images), duration=0.15)
            print(f"GIF saved to {gif_path}")
        except ImportError:
            print("imageio库未安装，无法自动合成GIF。可用 pip install imageio 安装。")

if __name__ == '__main__':
    # 批量可视化所有测试用例
    TEST_SUITE_A = [
        {"map_file": "test_maps_hard/set_A_for_candidates/case_standard_1.npz", "radius": 10, "step_limit": 500, "time_limit": 20},
        {"map_file": "test_maps_hard/set_A_for_candidates/case_standard_2.npz", "radius": 5, "step_limit": 2000, "time_limit": 30},
        {"map_file": "test_maps_hard/set_A_for_candidates/case_maze_1.npz", "radius": 10, "step_limit": 1500, "time_limit": 20},
        {"map_file": "test_maps_hard/set_A_for_candidates/case_maze_2.npz", "radius": 1000, "step_limit": 3000, "time_limit": 20},
        {"map_file": "test_maps_hard/set_A_for_candidates/case_maze_3.npz", "radius": 5, "step_limit": 8000, "time_limit": 40},
        {"map_file": "test_maps_hard/set_A_for_candidates/case_maze_4.npz", "radius": 20, "step_limit": 6000, "time_limit": 40},   
    ]
    for test_case in TEST_SUITE_A:
        run_dynamic_visualization(Planner, test_case, save_gif=True, vis_dir='dynamic_vis') 