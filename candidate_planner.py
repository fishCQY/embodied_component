import numpy as np
import random
from collections import deque
import heapq

class Planner:
    def __init__(self, map_shape: tuple[int, int], view_radius: int):
        self.known_map = np.full(map_shape, -1, dtype=int)
        self.view_radius = view_radius
        self.path_taken = []
        self.map_shape = map_shape
        
        # 存储边界点（frontier points）
        self.frontier_points = set()
        # 当前路径
        self.current_path = deque()
        # A*算法缓存
        self.a_star_cache = {}
        
    def update_knowledge(self, current_pos: tuple[int, int], local_view: np.ndarray):
        r = self.view_radius
        x, y = current_pos
        h, w = self.known_map.shape
        
        y_start, y_end = max(0, y - r), min(h, y + r + 1)
        x_start, x_end = max(0, x - r), min(w, x + r + 1)
        
        # 更新已知地图
        self.known_map[y_start:y_end, x_start:x_end] = local_view
        
        # 更新边界点
        self.update_frontier_points(current_pos)
    
    def update_frontier_points(self, current_pos: tuple[int, int]):
        """更新边界点集合（已知与未知的交界）"""
        r = self.view_radius
        x, y = current_pos
        h, w = self.known_map.shape
        
        y_start, y_end = max(0, y - r), min(h, y + r + 1)
        x_start, x_end = max(0, x - r), min(w, x + r + 1)
        
        # 清除当前视野内的边界点
        for py in range(y_start, y_end):
            for px in range(x_start, x_end):
                self.frontier_points.discard((px, py))
        
        # 添加新的边界点
        for py in range(y_start, y_end):
            for px in range(x_start, x_end):
                # 只处理自由空间单元格
                if self.known_map[py, px] != 0:
                    continue
                
                # 检查4邻域内是否有未知区域
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = px + dx, py + dy
                    if 0 <= ny < h and 0 <= nx < w:
                        if self.known_map[ny, nx] == -1:
                            self.frontier_points.add((px, py))
                            break
    
    def plan_next_step(self, current_pos: tuple[int, int], all_possible_targets: list[tuple[int, int]]) -> tuple[int, int]:
        if not self.path_taken or self.path_taken[-1] != current_pos:
            self.path_taken.append(current_pos)
        
        # 将目标列表转换为集合提高查找效率
        target_set = set(all_possible_targets)
        
        # 如果当前位置就是目标，返回当前位置
        if current_pos in target_set and self.known_map[current_pos[1], current_pos[0]] == 0:
            return current_pos
        
        # 如果有当前路径且未完成，继续沿着路径移动
        if self.current_path:
            next_pos = self.current_path.popleft()
            # 检查路径是否仍然有效（没有新发现的障碍）
            if self.is_valid_move(current_pos, next_pos):
                return next_pos
            else:
                # 路径无效，清除并重新规划
                self.current_path.clear()
        
        # 尝试规划到已知目标点的路径
        known_targets = [t for t in target_set 
                         if 0 <= t[1] < self.map_shape[0] and 
                         0 <= t[0] < self.map_shape[1] and 
                         self.known_map[t[1], t[0]] == 0]
        
        if known_targets:
            # 找到最近的目标点
            closest_target = min(known_targets, 
                                 key=lambda t: self.manhattan_distance(current_pos, t))
            
            # 使用A*规划到目标的路径
            path = self.a_star(current_pos, closest_target)
            if path:
                self.current_path = deque(path)
                if self.current_path:
                    return self.current_path.popleft()
        
        # 如果没有已知目标，尝试规划到边界点的路径
        if self.frontier_points:
            # 找到最近的边界点
            closest_frontier = min(self.frontier_points, 
                                  key=lambda p: self.manhattan_distance(current_pos, p))
            
            # 使用A*规划到边界点的路径
            path = self.a_star(current_pos, closest_frontier)
            if path:
                self.current_path = deque(path)
                if self.current_path:
                    return self.current_path.popleft()
        
        # 最后尝试随机合法移动
        return self.random_valid_move(current_pos)
    
    def a_star(self, start: tuple[int, int], goal: tuple[int, int]) -> list:
        """A*路径规划算法"""
        # 检查缓存
        cache_key = (start, goal)
        if cache_key in self.a_star_cache:
            return self.a_star_cache[cache_key][:]
        
        if start == goal:
            return []
        
        # 优先队列 (f_score, position)
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        # 记录路径
        came_from = {}
        
        # 代价函数
        g_score = {start: 0}
        f_score = {start: self.manhattan_distance(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # 重建路径
                path = []
                while current != start:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                # 缓存结果
                self.a_star_cache[cache_key] = path
                return path
            
            # 探索邻居
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # 检查边界
                if not (0 <= neighbor[1] < self.map_shape[0] and 
                        0 <= neighbor[0] < self.map_shape[1]):
                    continue
                
                # 检查是否可通过（自由空间或目标点）
                if (self.known_map[neighbor[1], neighbor[0]] != 0 and 
                    neighbor != goal):
                    continue
                
                # 计算新代价
                tentative_g = g_score[current] + 1
                
                # 如果找到更好的路径
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.manhattan_distance(neighbor, goal)
                    
                    # 添加到开放集
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # 无路径
        self.a_star_cache[cache_key] = []
        return None
    
    def random_valid_move(self, current_pos: tuple[int, int]) -> tuple[int, int]:
        """随机选择合法移动"""
        motions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(motions)
        
        for dx, dy in motions:
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            if self.is_valid_move(current_pos, next_pos):
                return next_pos
        
        # 无合法移动时保持原位
        return current_pos
    
    def is_valid_move(self, from_pos: tuple[int, int], to_pos: tuple[int, int]) -> bool:
        """检查移动是否合法"""
        x, y = to_pos
        h, w = self.map_shape
        
        # 检查边界
        if not (0 <= y < h and 0 <= x < w):
            return False
        
        # 检查是否移动到已知障碍物
        if self.known_map[y, x] == 1:
            return False
        
        # 检查是否为相邻移动
        dx = abs(from_pos[0] - to_pos[0])
        dy = abs(from_pos[1] - to_pos[1])
        return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)
    
    def manhattan_distance(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        """计算曼哈顿距离"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])