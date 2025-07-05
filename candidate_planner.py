import heapq
import numpy as np
from collections import deque
from scipy.ndimage import maximum_filter

class Planner:
    def __init__(self, map_shape: tuple[int, int], view_radius: int):
        self.known_map = np.full(map_shape, -1, dtype=int)
        self.view_radius = view_radius
        self.map_shape = map_shape
        self.motions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        self.frontier_map = np.zeros(map_shape, dtype=bool)
        self.explored_map = np.zeros(map_shape, dtype=bool)
        self.path_plan = deque()
        self.last_pos = None
        self.target_visible = False
        self.exploration_direction = None
        self.path_taken = []
        
        # 预定义检测核
        self.frontier_kernel = np.array([[0, 1, 0],
                                        [1, 0, 1],
                                        [0, 1, 0]], dtype=bool)
    
    def update_knowledge(self, current_pos: tuple[int, int], local_view: np.ndarray):
        r = self.view_radius
        x, y = current_pos
        h, w = self.map_shape
        
        # 计算当前视野范围
        y_start, y_end = max(0, y - r), min(h, y + r + 1)
        x_start, x_end = max(0, x - r), min(w, x + r + 1)
        
        # 更新已知地图
        self.known_map[y_start:y_end, x_start:x_end] = local_view
        self.explored_map[y_start:y_end, x_start:x_end] = True
        
        # 计算需要更新的区域（扩大一圈以覆盖可能受影响的邻居）
        y_low, y_high = max(0, y_start - 1), min(h, y_end + 1)
        x_low, x_high = max(0, x_start - 1), min(w, x_end + 1)
        
        # 重置受影响区域的前沿标记
        self.frontier_map[y_low:y_high, x_low:x_high] = False
        
        # 仅处理受影响区域
        region = self.known_map[y_low:y_high, x_low:x_high]
        is_free = (region == 0)
        
        # 创建未知区域掩码
        unknown_mask = (region == -1)
        
        # 使用卷积检测未知邻居 - 代替内部循环
        # 卷积结果中值大于0表示至少有一个未知邻居
        conv_result = maximum_filter(unknown_mask, footprint=self.frontier_kernel, mode='constant', cval=0)
        has_unknown_neighbor = conv_result & is_free
        
        # 将结果映射回前沿图
        self.frontier_map[y_low:y_high, x_low:x_high] |= has_unknown_neighbor

    def is_valid_move(self, pos):
        x, y = pos
        return (0 <= x < self.map_shape[1] and 
                0 <= y < self.map_shape[0] and 
                self.known_map[y, x] != 1)
    
    def get_safe_neighbors(self, current_pos):
        # 使用列表推导式+条件过滤代替显式循环
        return [(current_pos[0] + dx, current_pos[1] + dy) 
                for dx, dy in self.motions
                if self.is_valid_move((current_pos[0] + dx, current_pos[1] + dy))]

    # A*算法实现
    def a_star_search(self, start: tuple[int, int], goal: tuple[int, int]):
        """使用A*算法搜索最短路径"""
        if start == goal:
            return []
        
        # 开启集和关闭集
        open_set = []
        closed_set = set()
        # 路径记录
        came_from = {}
        
        # 起点到当前点的实际代价
        g_score = {start: 0}
        # 起点到终点的预估代价（实际代价+启发式值）
        f_score = {start: self.heuristic(start, goal)}
        
        heapq.heappush(open_set, (f_score[start], start))
        
        while open_set:
            # 获取当前最优节点
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # 构建路径（不包括起点）
                path = []
                while current != start:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
                
            closed_set.add(current)
            
            for neighbor in self.get_safe_neighbors(current):
                # 如果邻居已在关闭集中，跳过
                if neighbor in closed_set:
                    continue
                    
                # 计算新代价
                tentative_g = g_score[current] + 1
                
                if (neighbor not in g_score or tentative_g < g_score[neighbor]):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
        return []  # 未找到路径

    def heuristic(self, a: tuple[int, int], b: tuple[int, int]):
        """曼哈顿距离启发函数"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # BFS用于前沿点搜索（保持不变）
    def find_nearest_frontier_bfs(self, start_pos):
        """BFS查找最近的前沿点路径"""
        if self.frontier_map[start_pos[1], start_pos[0]]:
            return []
        
        queue = deque([start_pos])
        visited = {start_pos: None}
        
        while queue:
            current = queue.popleft()
            
            if self.frontier_map[current[1], current[0]]:
                path = []
                while current != start_pos:
                    path.append(current)
                    current = visited[current]
                return path[::-1]
            
            for neighbor in self.get_safe_neighbors(current):
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)
                    
        return []

    def plan_next_step(self, current_pos: tuple[int, int], all_possible_targets: list[tuple[int, int]]) -> tuple[int, int]:
        if self.path_plan:
            next_step = self.path_plan.popleft()
            if self.is_valid_move(next_step):
                self.last_pos = current_pos
                return next_step
        
        self.target_visible = False
        reachable_targets = [t for t in all_possible_targets 
                            if self.is_valid_move(t) and 
                            self.known_map[t[1], t[0]] == 0]
        
        if reachable_targets:
            self.target_visible = True
            for target in reachable_targets:
                if target[1] == current_pos[1]:
                    step_x = target[0] - current_pos[0]
                    if step_x != 0:
                        next_step = (current_pos[0] + (1 if step_x > 0 else -1), current_pos[1])
                        if self.is_valid_move(next_step):
                            self.last_pos = current_pos
                            return next_step
                
                if target[0] == current_pos[0]:
                    step_y = target[1] - current_pos[1]
                    if step_y != 0:
                        next_step = (current_pos[0], current_pos[1] + (1 if step_y > 0 else -1))
                        if self.is_valid_move(next_step):
                            self.last_pos = current_pos
                            return next_step
            
            closest_target = min(reachable_targets, 
                                key=lambda t: abs(t[0]-current_pos[0]) + abs(t[1]-current_pos[1]))
            
            # 使用A*算法规划到目标点的路径
            path = self.a_star_search(current_pos, closest_target)
            if path:
                self.path_plan = deque(path)
                return self.plan_next_step(current_pos, all_possible_targets)
        
        # 如果找不到目标点路径，尝试前沿点搜索
        frontier_path = self.find_nearest_frontier_bfs(current_pos)
        if frontier_path:
            self.path_plan = deque(frontier_path)
            return self.plan_next_step(current_pos, all_possible_targets)
        
        # 最后策略：移动到未知区域或最少探索的邻居
        safe_moves = self.get_safe_neighbors(current_pos)
        if safe_moves:
            for move in safe_moves:
                if self.known_map[move[1], move[0]] == -1:
                    self.last_pos = current_pos
                    return move
            
            best_move = None
            min_explored = float('inf')
            for move in safe_moves:
                unexplored_count = sum(1 for dx, dy in self.motions
                                       for nx, ny in [(move[0]+dx, move[1]+dy)]
                                       if (0 <= nx < self.map_shape[1] and
                                           0 <= ny < self.map_shape[0] and
                                           not self.explored_map[ny, nx]))
                
                if unexplored_count < min_explored or best_move is None:
                    best_move = move
                    min_explored = unexplored_count
            
            if best_move:
                self.last_pos = current_pos
                return best_move
        
        # 保底策略：任意可行方向
        for dx, dy in self.motions:
            next_step = (current_pos[0] + dx, current_pos[1] + dy)
            if self.is_valid_move(next_step):
                self.last_pos = current_pos
                return next_step
        
        return current_pos