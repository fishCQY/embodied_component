import numpy as np
import random
import heapq
from collections import deque

class Planner:
    def __init__(self, map_shape: tuple[int, int], view_radius: int):
        self.known_map = np.full(map_shape, -1, dtype=int)
        self.view_radius = view_radius
        self.path_taken = []
        self.map_shape = map_shape
        self.frontier = set()
        self.prioritized_frontier = []  # 按优先级排序的边界点
        self.current_path = []  # 当前规划的路径
        self.last_target = None  # 上次规划的目标
        
    def update_knowledge(self, current_pos: tuple[int, int], local_view: np.ndarray):
        r = self.view_radius
        cx, cy = current_pos
        h, w = self.map_shape
        
        y_start, y_end = max(0, cy - r), min(h, cy + r + 1)
        x_start, x_end = max(0, cx - r), min(w, cx + r + 1)
        
        self.known_map[y_start:y_end, x_start:x_end] = local_view
        
        updated_region = [
            max(0, y_start-1), min(h, y_end+1),
            max(0, x_start-1), min(w, x_end+1)
        ]
        
        # 清除原有边界点
        for point in list(self.frontier):
            x, y = point
            if updated_region[0] <= y < updated_region[1] and \
               updated_region[2] <= x < updated_region[3]:
                self.frontier.discard(point)
        
        # 添加新边界点并计算优先级
        new_frontier = set()
        for y in range(updated_region[0], updated_region[1]):
            for x in range(updated_region[2], updated_region[3]):
                if self.known_map[y, x] != 0:
                    continue
                
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= ny < h and 0 <= nx < w and 
                        self.known_map[ny, nx] == -1):
                        # 计算到当前位置的距离作为优先级
                        dist = abs(x - cx) + abs(y - cy)
                        new_frontier.add((dist, (x, y)))
                        self.frontier.add((x, y))
                        break
        
        # 按距离排序边界点
        self.prioritized_frontier = sorted(new_frontier)
    # 在Planner类中添加以下方法
    def _get_obstacle_distance(self, x, y, search_radius=5):
        """计算给定位置到最近障碍物的曼哈顿距离"""
        min_dist = float('inf')
        for dy in range(-search_radius, search_radius+1):
            for dx in range(-search_radius, search_radius+1):
                nx, ny = x + dx, y + dy
                if not (0 <= ny < self.map_shape[0] and 0 <= nx < self.map_shape[1]):
                    continue
                if self.known_map[ny, nx] == 1:  # 障碍物
                    dist = abs(dx) + abs(dy)  # 曼哈顿距离
                    if dist < min_dist:
                        min_dist = dist
        return min_dist if min_dist < float('inf') else search_radius * 2

    def _calculate_heuristic(self, x, y, reachable_targets):
        """新型启发函数：结合目标距离和避障因素的加权启发值"""
        # 1. 计算到目标的基础距离值
        if reachable_targets:
            target_dist = min(abs(x - tx) + abs(y - ty) for tx, ty in reachable_targets)
        elif self.prioritized_frontier:
            target_dist = self.prioritized_frontier[0][0]
        else:
            target_dist = 0
        
        # 2. 计算避障因子（到最近障碍物的距离）
        obstacle_dist = self._get_obstacle_distance(x, y)
        
        # 3. 动态权重调整 - 在复杂区域提高避障权重
        # 基于当前位置的环境特性动态调整权重
        local_density = self._get_local_obstacle_density(x, y)
        
        # 避障权重：在狭窄区域赋予更高权重
        # 目标权重：在开阔区域赋予更高权重
        if local_density > 0.4:  # 狭窄迷宫区域
            obstacle_weight = 0.7
            target_weight = 0.3
        else:  # 开阔区域
            obstacle_weight = 0.3
            target_weight = 0.7
        
        # 4. 综合启发值：组合避障距离和目标距离
        # 注意：避障距离越大越好（离障碍物越远），所以使用加权和
        return target_weight * target_dist - obstacle_weight * obstacle_dist

    def _get_local_obstacle_density(self, x, y):
        """计算局部障碍物密度（用于动态调整权重）"""
        sample_radius = 3
        total_cells = 0
        obstacle_count = 0
        
        for dy in range(-sample_radius, sample_radius+1):
            for dx in range(-sample_radius, sample_radius+1):
                nx, ny = x + dx, y + dy
                if not (0 <= ny < self.map_shape[0] and 0 <= nx < self.map_shape[1]):
                    continue
                total_cells += 1
                if self.known_map[ny, nx] == 1:  # 障碍物
                    obstacle_count += 1
                    
        return obstacle_count / total_cells if total_cells > 0 else 0
    def plan_next_step(self, current_pos: tuple[int, int], 
                      all_possible_targets: list[tuple[int, int]]) -> tuple[int, int]:
        if not self.path_taken or self.path_taken[-1] != current_pos:
            self.path_taken.append(current_pos)
        
        # 过滤可达目标
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
        
        # 尝试重用现有路径
        if self.last_target == all_possible_targets and self.current_path:
            next_step = self.current_path.pop(0)
            nx, ny = next_step
            if (0 <= ny < self.map_shape[0] and 
                0 <= nx < self.map_shape[1] and
                self.known_map[ny, nx] == 0):
                return next_step
        
        # 检查邻近点
        cx, cy = current_pos
        motions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(motions)
        
        for dx, dy in motions:
            nx, ny = cx + dx, cy + dy
            neighbor = (nx, ny)
            
            if not (0 <= ny < self.map_shape[0] and 0 <= nx < self.map_shape[1]):
                continue
                
            # 发现目标
            if neighbor in reachable_targets:
                return neighbor
                
            # 发现边界点
            if self.known_map[ny, nx] == 0 and (nx, ny) in self.frontier:
                return neighbor
        
        # A*算法搜索
        open_set = []
        heapq.heappush(open_set, (0, current_pos))
        g_score = {current_pos: 0}
        parent = {current_pos: None}
        
        found_target = None
        found_frontier = None
        
        while open_set:
            _, (x, y) = heapq.heappop(open_set)
            pos = (x, y)
            
            # 找到目标点
            if pos in reachable_targets:
                found_target = pos
                break
                
            # 找到边界点
            if found_frontier is None and pos in self.frontier:
                found_frontier = pos
                
            # 扩展邻居
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                
                if not (0 <= ny < self.map_shape[0] and 0 <= nx < self.map_shape[1]):
                    continue
                    
                if self.known_map[ny, nx] != 0:
                    continue
                    
                tentative_g = g_score.get(pos, float('inf')) + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    
                    # 启发函数：到最近目标的曼哈顿距离
                    if reachable_targets:
                        h_score = min(abs(nx - tx) + abs(ny - ty) for tx, ty in reachable_targets)
                    else:
                        # 如果没有可达目标，使用到最近边界点的距离
                        if self.prioritized_frontier:
                            h_score = self.prioritized_frontier[0][0]
                        else:
                            h_score = 0
                    
                    f_score = tentative_g + h_score
                    heapq.heappush(open_set, (f_score, neighbor))
                    parent[neighbor] = pos
        
        # 回溯路径
        def reconstruct_path(target):
            path = []
            current = target
            while current != current_pos:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path
        
        # 优先返回目标点路径
        if found_target:
            self.current_path = reconstruct_path(found_target)
            self.last_target = all_possible_targets
            return self.current_path.pop(0) if self.current_path else current_pos
            
        # 其次返回边界点路径
        if found_frontier:
            self.current_path = reconstruct_path(found_frontier)
            self.last_target = all_possible_targets
            return self.current_path.pop(0) if self.current_path else current_pos
        
        # 尝试使用优先边界点
        if self.prioritized_frontier:
            _, nearest_frontier = self.prioritized_frontier[0]
            # 简单路径规划到最近边界点
            motions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(motions)
            for dx, dy in motions:
                nx, ny = cx + dx, cy + dy
                if (0 <= ny < self.map_shape[0] and 
                    0 <= nx < self.map_shape[1] and
                    self.known_map[ny, nx] == 0):
                    return (nx, ny)
        
        # 最终回退：随机选择有效移动方向
        for dx, dy in motions:
            nx, ny = cx + dx, cy + dy
            if (0 <= ny < self.map_shape[0] and 
                0 <= nx < self.map_shape[1] and
                self.known_map[ny, nx] != 1):
                return (nx, ny)
        
        return current_pos