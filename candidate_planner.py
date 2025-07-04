import numpy as np
import random
import math
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
        # 存储目标点
        self.targets = set()
        # 当前路径
        self.current_path = deque()
        
        # 边界点搜索参数
        self.kernel_size = max(3, view_radius // 2)  # 边界检测核大小
        self.last_frontier = None  # 上次选择的边界点
        
        # 快速搜索数据结构
        self.unexplored_mask = np.ones(map_shape, dtype=bool)  # 修复：使用实例变量map_shape
        
        # A*算法缓存
        self.a_star_cache = {}
        
    def update_knowledge(self, current_pos: tuple[int, int], local_view: np.ndarray):
        r = self.view_radius
        x, y = current_pos
        h, w = self.map_shape  # 使用self.map_shape
        
        y_start, y_end = max(0, y - r), min(h, y + r + 1)
        x_start, x_end = max(0, x - r), min(w, x + r + 1)
        
        # 保存旧地图状态用于边界点更新
        old_map = self.known_map.copy()
        
        # 更新已知地图
        self.known_map[y_start:y_end, x_start:x_end] = local_view
        
        # 更新未探索区域掩码
        self.unexplored_mask[y_start:y_end, x_start:x_end] = False
        
        # 更新边界点
        self.update_frontier_points(old_map, y_start, y_end, x_start, x_end)
    
    def update_frontier_points(self, old_map, y_start, y_end, x_start, x_end):
        """高效更新边界点集合（已知与未知的交界）"""
        # 清除当前视野内的边界点
        for py in range(y_start, y_end):
            for px in range(x_start, x_end):
                self.frontier_points.discard((px, py))
        
        # 添加新的边界点（仅检查当前视野内的新探索单元）
        for py in range(y_start, y_end):
            for px in range(x_start, x_end):
                # 只处理新探索的空地单元格
                if old_map[py, px] != self.known_map[py, px] and self.known_map[py, px] == 0:
                    # 检查8邻域内是否有未知区域
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = px + dx, py + dy
                            if 0 <= ny < self.map_shape[0] and 0 <= nx < self.map_shape[1]:
                                if self.known_map[ny, nx] == -1:
                                    self.frontier_points.add((px, py))
                                    break
                        else:
                            continue
                        break
    
    def plan_next_step(self, current_pos: tuple[int, int], all_possible_targets: list[tuple[int, int]]) -> tuple[int, int]:
        if not self.path_taken or self.path_taken[-1] != current_pos:
            self.path_taken.append(current_pos)
        
        # 更新目标集合
        self.targets = set(all_possible_targets)
        
        # 如果当前位置是目标，直接返回
        if current_pos in self.targets and self.known_map[current_pos[1], current_pos[0]] == 0:
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
        known_targets = [t for t in self.targets 
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
            # 高效查找最近的边界点
            closest_frontier = self.find_closest_frontier_efficient(current_pos)
            if closest_frontier:
                # 使用A*规划到边界点的路径
                path = self.a_star(current_pos, closest_frontier)
                if path:
                    self.current_path = deque(path)
                    if self.current_path:
                        return self.current_path.popleft()
        
        # 最后尝试随机合法移动（朝向未探索区域）
        return self.directed_random_move(current_pos)
    
    def find_closest_frontier_efficient(self, current_pos: tuple[int, int]) -> tuple[int, int]:
        """高效查找最近的边界点（三步策略）"""
        if not self.frontier_points:
            return None
        
        # 1. 如果有上次的边界点且仍存在，优先返回
        if self.last_frontier and self.last_frontier in self.frontier_points:
            return self.last_frontier
        
        # 2. 使用最近探索历史寻找边界点（常数时间）
        closest = self.find_near_frontier_from_history(current_pos)
        if closest:
            self.last_frontier = closest
            return closest
        
        # 3. 使用距离启发式选择（O(N)但N很小）
        closest = min(self.frontier_points, key=lambda p: self.manhattan_distance(current_pos, p))
        self.last_frontier = closest
        return closest
    
    def find_near_frontier_from_history(self, current_pos):
        """从最近的探索历史中查找边界点（常数时间）"""
        # 检查最近的10个路径点
        history_points = self.path_taken[-10:] if len(self.path_taken) > 10 else self.path_taken
        
        # 为每个点检查周围2x2区域
        for pos in history_points:
            x, y = pos
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    point = (x + dx, y + dy)
                    if point in self.frontier_points:
                        return point
        return None
    
    def a_star(self, start: tuple[int, int], goal: tuple[int, int]) -> list:
        """带缓存的A*路径规划算法"""
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
    
    def directed_random_move(self, current_pos: tuple[int, int]) -> tuple[int, int]:
        """智能随机移动：倾向探索方向"""
        motions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        valid_moves = []
        
        # 收集所有有效移动
        for dx, dy in motions:
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            if self.is_valid_move(current_pos, next_pos):
                valid_moves.append(next_pos)
        
        if not valid_moves:
            return current_pos
        
        # 优先选择未知区域方向
        for move in valid_moves:
            x, y = move
            if (0 <= y < self.map_shape[0] and 0 <= x < self.map_shape[1] and
                self.known_map[y, x] == -1):
                return move
        
        # 其次选择已知自由空间
        for move in valid_moves:
            x, y = move
            if (0 <= y < self.map_shape[0] and 0 <= x < self.map_shape[1] and
                self.known_map[y, x] == 0):
                return move
        
        # 最后选择任意有效移动（应有但在上述条件未覆盖）
        return valid_moves[0]
    
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