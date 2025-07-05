import numpy as np
import random
import heapq
from collections import deque

class Planner:
    def __init__(self, map_shape: tuple[int, int], view_radius: int):
        self.known_map = np.full(map_shape, -1, dtype=int)  # -1代表未知区域，0代表空地，1代表障碍物
        self.view_radius = view_radius
        self.path_taken = []
        self.map_shape = map_shape
        
    def update_knowledge(self, current_pos: tuple[int, int], local_view: np.ndarray):
        r = self.view_radius
        x, y = current_pos
        h, w = self.known_map.shape
        
        y_start, y_end = max(0, y - r), min(h, y + r + 1)
        x_start, x_end = max(0, x - r), min(w, x + r + 1)
        
        self.known_map[y_start:y_end, x_start:x_end] = local_view

    def find_boundary_points(self, current_pos: tuple[int, int]):
        """寻找边界点，即未知区域的边界"""
        boundary_points = []
        x, y = current_pos
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= ny < self.map_shape[0] and 0 <= nx < self.map_shape[1]:
                if self.known_map[ny, nx] == -1:  # 发现未知区域
                    boundary_points.append((nx, ny))
        return boundary_points

    def a_star(self, start: tuple[int, int], goal: tuple[int, int]):
        """A*算法路径规划，比BFS更快找到最短路径"""
        if start == goal:
            return [start]
            
        # 初始化开放列表（优先队列）
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        # 跟踪路径
        came_from = {}
        
        # 起点到各点的实际代价
        g_score = {start: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            # 如果到达目标点，重建路径
            if current == goal:
                path = [current]
                while current != start:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            
            cx, cy = current
            # 探索四个方向
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)
                
                # 边界检查
                if not (0 <= ny < self.map_shape[0] and 0 <= nx < self.map_shape[1]):
                    continue
                
                # 跳过障碍物
                if self.known_map[ny, nx] == 1:
                    continue
                
                # 计算新的代价
                new_g = g_score[current] + 1
                
                # 如果这是更好的路径，更新路径
                if neighbor not in g_score or new_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = new_g
                    # 计算启发式代价（曼哈顿距离）
                    h_score = abs(nx - goal[0]) + abs(ny - goal[1])
                    f_score = new_g + h_score
                    heapq.heappush(open_set, (f_score, neighbor))
        
        return None  # 没有找到路径

    def plan_next_step(self, current_pos: tuple[int, int], all_possible_targets: list[tuple[int, int]]) -> tuple[int, int]:
        if not self.path_taken or self.path_taken[-1] != current_pos:
            self.path_taken.append(current_pos)
        
        # 将目标列表转换为集合提高查找效率
        target_set = set(all_possible_targets)
        
        # 如果当前位置就是目标，返回当前位置（模拟器会处理到达目标）
        if current_pos in target_set and self.known_map[current_pos[1], current_pos[0]] == 0:
            return current_pos
        
        # 首先检查是否有已知目标点可以使用A*算法
        for target in all_possible_targets:
            tx, ty = target
            if (0 <= ty < self.map_shape[0] and 
                0 <= tx < self.map_shape[1] and
                self.known_map[ty, tx] == 0):  # 确保目标点是已知空地
                
                # 使用A*算法规划到目标点的路径
                path = self.a_star(current_pos, target)
                if path and len(path) > 1:
                    return path[1]  # 返回路径的第一步
        
        # 没有找到已知目标点，执行BFS进行探索
        queue = deque([current_pos])
        visited = {current_pos}
        parent = {current_pos: None}
        
        found_target = None
        found_frontier = None
        
        # BFS搜索最近的目标或边界
        while queue and not found_target:  # 如果找到目标就提前退出
            x, y = queue.popleft()
            
            # 检查是否找到目标
            if (x, y) in target_set and self.known_map[y, x] == 0:
                found_target = (x, y)
                break
            
            # 检查是否是边界点（未知区域的边界）
            if found_frontier is None:
                boundary_points = self.find_boundary_points((x, y))
                if boundary_points:
                    found_frontier = (x, y)
            
            # 扩展邻居节点
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)
                
                # 检查边界和可通行性
                if (0 <= ny < self.map_shape[0] and 
                    0 <= nx < self.map_shape[1] and
                    self.known_map[ny, nx] == 0 and 
                    next_pos not in visited):
                    
                    visited.add(next_pos)
                    queue.append(next_pos)
                    parent[next_pos] = (x, y)
        
        # 回溯路径找到第一步移动的函数
        def find_first_step(target):
            step = target
            while parent[step] != current_pos:
                step = parent[step]
            return step
        
        # 优先处理找到的目标（使用A*算法加速）
        if found_target:
            path = self.a_star(current_pos, found_target)
            if path and len(path) > 1:
                return path[1]  # 使用A*返回下一步
            else:
                return find_first_step(found_target)  # 如果A*失败则回退到BFS方法
        
        # 其次处理边界点
        if found_frontier:
            return find_first_step(found_frontier)
        
        # 最后尝试随机合法移动
        motions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(motions)
        for dx, dy in motions:
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            nx, ny = next_pos
            if (0 <= ny < self.map_shape[0] and 
                0 <= nx < self.map_shape[1] and
                self.known_map[ny, nx] != 1):  # 避免已知障碍
                return next_pos
        
        # 无合法移动时保持原位
        return current_pos