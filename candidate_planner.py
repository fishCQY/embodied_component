import numpy as np
import random
from collections import deque

class Planner:
    def __init__(self, map_shape: tuple[int, int], view_radius: int):
        self.known_map = np.full(map_shape, -1, dtype=int)  # -1:未知, 0:空地, 1:障碍
        self.view_radius = view_radius
        self.path_taken = []
        self.map_shape = map_shape
        self.frontier = set()  # 维护边界点集合（空间换时间）
        
    def update_knowledge(self, current_pos: tuple[int, int], local_view: np.ndarray):
        r = self.view_radius
        cx, cy = current_pos
        h, w = self.map_shape
        
        # 计算局部视图在全局地图中的边界
        y_start, y_end = max(0, cy - r), min(h, cy + r + 1)
        x_start, x_end = max(0, cx - r), min(w, cx + r + 1)
        
        # 更新全局地图
        self.known_map[y_start:y_end, x_start:x_end] = local_view
        
        # 更新边界点：只更新受影响的区域（高效的关键）
        updated_region = [
            max(0, y_start-1), min(h, y_end+1),
            max(0, x_start-1), min(w, x_end+1)
        ]
        
        # 清除原有边界点（仅更新区域内的）
        for point in list(self.frontier):
            x, y = point
            if updated_region[0] <= y < updated_region[1] and \
               updated_region[2] <= x < updated_region[3]:
                self.frontier.discard(point)
        
        # 添加新边界点（仅更新区域内）
        for y in range(updated_region[0], updated_region[1]):
            for x in range(updated_region[2], updated_region[3]):
                if self.known_map[y, x] != 0:
                    continue  # 只考虑自由区域
                
                # 检查四个方向是否有未知区域
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= ny < h and 0 <= nx < w and 
                        self.known_map[ny, nx] == -1):
                        self.frontier.add((x, y))
                        break

    def plan_next_step(self, current_pos: tuple[int, int], 
                      all_possible_targets: list[tuple[int, int]]) -> tuple[int, int]:
        # 记录路径
        if not self.path_taken or self.path_taken[-1] != current_pos:
            self.path_taken.append(current_pos)
        
        # 高效目标点检查：使用集合加快查找
        reachable_targets = set()
        for target in all_possible_targets:
            tx, ty = target
            if (0 <= ty < self.map_shape[0] and 
                0 <= tx < self.map_shape[1] and
                self.known_map[ty, tx] == 0):
                reachable_targets.add(target)
        
        # 当前位置即目标
        if current_pos in reachable_targets:
            return current_pos
        
        # 优化1：直接检查邻近点是否为目标或边界点
        cx, cy = current_pos
        nearest_target = None
        motions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(motions)
        
        for dx, dy in motions:
            nx, ny = cx + dx, cy + dy
            neighbor = (nx, ny)
            
            # 边界检查
            if not (0 <= ny < self.map_shape[0] and 0 <= nx < self.map_shape[1]):
                continue
                
            # 发现目标
            if neighbor in reachable_targets:
                return neighbor
                
            # 发现边界点（直接可达）
            if self.known_map[ny, nx] == 0 and neighbor in self.frontier:
                nearest_target = neighbor  # 记录但不立即返回，继续检查剩余方向
        
        if nearest_target:
            return nearest_target
        
        # 优化2：多源BFS同时搜索目标和边界
        queue = deque([current_pos])
        visited = {current_pos}
        parent = {current_pos: None}
        
        found_target = None
        found_frontier = None
        
        while queue:
            x, y = queue.popleft()
            pos = (x, y)
            
            # 找到目标点
            if not found_target and pos in reachable_targets:
                found_target = pos
            
            # 找到边界点
            if not found_frontier and pos in self.frontier:
                found_frontier = pos
            
            # 已找到两种目标则退出（优先满足任务目标）
            if found_target or found_frontier:
                # 如果已经找到目标点，不需要继续寻找边界点
                if found_target: 
                    break
                # 如果当前队列中有目标点则继续寻找
                if any(p in reachable_targets for p in queue):
                    continue
                break
                
            # 扩展邻居
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                
                # 边界检查
                if not (0 <= ny < self.map_shape[0] and 0 <= nx < self.map_shape[1]):
                    continue
                
                # 访问检查：未访问且可通行
                if (self.known_map[ny, nx] == 0 and 
                    neighbor not in visited):
                    
                    visited.add(neighbor)
                    queue.append(neighbor)
                    parent[neighbor] = pos
        
        # 回溯路径找到第一步移动
        def find_first_step(target):
            step = target
            while parent.get(step, current_pos) != current_pos:
                step = parent[step]
            return step
        
        # 优先返回目标点路径
        if found_target:
            return find_first_step(found_target)
            
        # 其次返回边界点路径
        if found_frontier:
            return find_first_step(found_frontier)
        
        # 最终回退：随机选择有效移动方向
        for dx, dy in motions:
            nx, ny = cx + dx, cy + dy
            if (0 <= ny < self.map_shape[0] and 
                0 <= nx < self.map_shape[1] and
                self.known_map[ny, nx] != 1):  # 避免障碍物
                return (nx, ny)
        
        # 无合法移动时保持原位
        return current_pos