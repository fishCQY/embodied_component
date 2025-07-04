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
        
        # RRT参数
        self.rrt_step_size = max(3, view_radius // 2)
        self.rrt_max_nodes = 500
        
    def update_knowledge(self, current_pos: tuple[int, int], local_view: np.ndarray):
        r = self.view_radius
        x, y = current_pos
        h, w = self.known_map.shape
        
        y_start, y_end = max(0, y - r), min(h, y + r + 1)
        x_start, x_end = max(0, x - r), min(w, x + r + 1)
        
        # 保存旧地图状态用于边界点更新
        old_map = self.known_map.copy()
        
        # 更新已知地图
        self.known_map[y_start:y_end, x_start:x_end] = local_view
        
        # 更新边界点
        self.update_frontier_points(old_map, current_pos)
    
    def update_frontier_points(self, old_map: np.ndarray, current_pos: tuple[int, int]):
        """更新边界点集合（已知与未知的交界）"""
        # 清除当前视野内的边界点（因为它们可能不再是边界）
        r = self.view_radius
        x, y = current_pos
        h, w = self.known_map.shape
        
        y_start, y_end = max(0, y - r), min(h, y + r + 1)
        x_start, x_end = max(0, x - r), min(w, x + r + 1)
        
        # 移除当前视野内的边界点
        for py in range(y_start, y_end):
            for px in range(x_start, x_end):
                if (px, py) in self.frontier_points:
                    self.frontier_points.discard((px, py))
        
        # 添加新的边界点
        for y in range(h):
            for x in range(w):
                # 只检查新探索的区域
                if old_map[y, x] == self.known_map[y, x]:
                    continue
                    
                # 如果是自由空间，检查其邻居是否有未知区域
                if self.known_map[y, x] == 0:
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= ny < h and 0 <= nx < w:
                            if self.known_map[ny, nx] == -1:  # 未知邻居
                                self.frontier_points.add((x, y))
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
            # 使用RRT找到最近的边界点
            closest_frontier = self.find_closest_frontier_rrt(current_pos)
            if closest_frontier:
                # 使用A*规划到边界点的路径
                path = self.a_star(current_pos, closest_frontier)
                if path:
                    self.current_path = deque(path)
                    if self.current_path:
                        return self.current_path.popleft()
        
        # 最后尝试随机合法移动
        return self.random_valid_move(current_pos)
    
    def find_closest_frontier_rrt(self, start_pos: tuple[int, int]) -> tuple[int, int]:
        """使用RRT算法快速找到最近的边界点"""
        if not self.frontier_points:
            return None
        
        # 创建RRT树
        tree = {start_pos: None}  # 节点: 父节点
        nodes = [start_pos]
        
        for _ in range(self.rrt_max_nodes):
            # 随机采样点
            if random.random() < 0.3:  # 30%概率采样边界点
                rand_point = random.choice(list(self.frontier_points))
            else:
                rand_point = (
                    random.randint(0, self.map_shape[1] - 1),
                    random.randint(0, self.map_shape[0] - 1)
                )
            
            # 找到最近的树节点
            nearest_node = min(nodes, key=lambda n: self.euclidean_distance(n, rand_point))
            
            # 向随机点方向移动一步
            direction = self.get_direction(nearest_node, rand_point)
            new_node = (
                nearest_node[0] + int(direction[0] * self.rrt_step_size),
                nearest_node[1] + int(direction[1] * self.rrt_step_size)
            )
            
            # 确保新节点在地图范围内
            new_node = (
                max(0, min(self.map_shape[1] - 1, new_node[0])),
                max(0, min(self.map_shape[0] - 1, new_node[1]))
            )
            
            # 检查路径是否有效（无已知障碍）
            if self.is_valid_path(nearest_node, new_node):
                # 添加到树
                tree[new_node] = nearest_node
                nodes.append(new_node)
                
                # 检查是否到达边界点
                if new_node in self.frontier_points:
                    return new_node
        
        # 如果没有找到边界点，返回最近的边界点
        if self.frontier_points:
            return min(self.frontier_points, key=lambda p: self.manhattan_distance(start_pos, p))
        
        return None
    
    def a_star(self, start: tuple[int, int], goal: tuple[int, int]) -> list:
        """A*路径规划算法"""
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
    
    def is_valid_path(self, from_pos: tuple[int, int], to_pos: tuple[int, int]) -> bool:
        """检查两点之间的直线路径是否有效（无已知障碍）"""
        # 使用Bresenham算法检查直线路径
        x0, y0 = from_pos
        x1, y1 = to_pos
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while x0 != x1 or y0 != y1:
            # 检查当前点是否已知障碍
            if (0 <= y0 < self.map_shape[0] and 
                0 <= x0 < self.map_shape[1] and 
                self.known_map[y0, x0] == 1):
                return False
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return True
    
    def get_direction(self, from_pos: tuple[int, int], to_pos: tuple[int, int]) -> tuple[float, float]:
        """获取从from_pos到to_pos的单位方向向量"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        dist = max(1e-5, math.sqrt(dx*dx + dy*dy))
        return (dx / dist, dy / dist)
    
    def manhattan_distance(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        """计算曼哈顿距离"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def euclidean_distance(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        """计算欧几里得距离"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)