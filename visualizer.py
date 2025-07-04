import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

def visualize_exploration_step(title: str, filename: str, true_grid_map: np.ndarray, known_map: np.ndarray, path_taken: list, current_pos: tuple, targets: set):
    """
    可视化探索过程，并将结果保存到文件中。
    """
    display_map = np.copy(true_grid_map).astype(float)
    
    # 将未知区域变暗
    display_map[known_map == -1] = -1 # -1 代表未知

    cmap = mcolors.ListedColormap(['#1a1a1a', '#FFFFFF', '#333333']) # 未知，空地，障碍物
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(display_map, cmap=cmap, norm=norm)

    # 绘制已走过的路径
    if path_taken:
        path_arr = np.array(path_taken)
        ax.plot(path_arr[:, 0], path_arr[:, 1], color='#2196F3', linewidth=2.5, alpha=0.8, label='Path Taken')

    # 用大五角星标记起点、当前位置和终点
    if path_taken:
        start_pos = path_taken[0]
        ax.scatter(start_pos[0], start_pos[1], s=400, c='#4CAF50', marker='*', zorder=10, label='Start')

    ax.scatter(current_pos[0], current_pos[1], s=500, c='#FFC107', marker='*', zorder=11, label='Current')

    if targets:
        for pos in targets:
            # 只显示已发现的目标
            if known_map[pos[1], pos[0]] != -1:
                 ax.scatter(pos[0], pos[1], s=400, c='#F44336', marker='*', zorder=10, label='Target')

    # 绘制网格
    ax.set_xticks(np.arange(-.5, true_grid_map.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, true_grid_map.shape[0], 1), minor=True)
    ax.grid(which="minor", color="k", linestyle='-', linewidth=0.5, alpha=0.05)
    
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.title(title, fontsize=16)
    plt.legend()
    
    # --- 保存文件 ---
    output_dir = "vis"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 清理文件名并保存
    safe_filename = "".join([c for c in filename if c.isalpha() or c.isdigit() or c in ('_','-')]).rstrip()
    save_path = os.path.join(output_dir, f"{safe_filename}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig) # 关闭图形，释放内存
    print(f"  -> 可视化结果已保存至: {save_path}")
