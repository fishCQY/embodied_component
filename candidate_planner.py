import numpy as np
from collections import deque

class Planner:
    def __init__(self, map_shape: tuple[int, int], view_radius: int):
        # 初始化地图知识
        self.known_map = np.full(map_shape, -1, dtype=int)  # -1未知，0自由空间，1障碍物
        self.view_radius = view_radius
        self.map_shape = map_shape
        self.motions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 上下左右
        
        # 探索状态跟踪
        self.frontier_map = np.zeros(map_shape, dtype=bool)  # 标记前沿点
        self.explored_map = np.zeros(map_shape, dtype=bool)  # 标记已探索区域
        self.path_plan = deque()  # 当前路径计划
        self.last_pos = None  # 上一位置，用于检测移动
        
        # 状态标志
        self.target_visible = False  # 目标是否可见
        self.exploration_direction = None  # 当前探索方向
        self.path_taken = []  # 记录路径    
    
    def update_knowledge(self, current_pos: tuple[int, int], local_view: np.ndarray):
        """更新机器人的环境知识"""
        r = self.view_radius
        x, y = current_pos
        h, w = self.known_map.shape
        
        # 计算当前视野范围
        y_start, y_end = max(0, y - r), min(h, y + r + 1)
        x_start, x_end = max(0, x - r), min(w, x + r + 1)
        
        # 更新已知地图
        self.known_map[y_start:y_end, x_start:x_end] = local_view
        self.explored_map[y_start:y_end, x_start:x_end] = True
        
        # 更新前沿点
        self.frontier_map.fill(False)  # 重置前沿图
        for y in range(1, h-1):
            for x in range(1, w-1):
                if self.known_map[y, x] == 0:  # 自由空间才可能是前沿
                    # 检查是否有相邻的未知区域
                    for dx, dy in self.motions:
                        nx, ny = x + dx, y + dy
                        if (0 <= ny < h and 0 <= nx < w and 
                            self.known_map[ny, nx] == -1):
                            self.frontier_map[y, x] = True
                            break

    def is_valid_move(self, pos):
        """检查移动位置是否合法"""
        x, y = pos
        # 检查是否在边界内且不是已知障碍物
        return (0 <= x < self.map_shape[1] and 
                0 <= y < self.map_shape[0] and 
                self.known_map[y, x] != 1)
    
    def get_safe_neighbors(self, current_pos):
        """获取所有安全的相邻位置（确保不会非法移动）"""
        safe_moves = []
        for dx, dy in self.motions:
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            if self.is_valid_move(next_pos):
                safe_moves.append(next_pos)
        return safe_moves

    def find_nearest_frontier_bfs(self, start_pos):
        """BFS查找最近的前沿点路径"""
        # 如果当前位置就在前沿点，直接返回
        if self.frontier_map[start_pos[1], start_pos[0]]:
            return []
        
        # BFS设置
        queue = deque([start_pos])
        visited = {start_pos: None}  # 存储路径回溯
        
        while queue:
            current = queue.popleft()
            
            # 到达前沿点，重建路径
            if self.frontier_map[current[1], current[0]]:
                path = []
                while current != start_pos:
                    path.append(current)
                    current = visited[current]
                return path[::-1]  # 反转路径：从起点到目标
            
            # 探索所有安全邻居
            for neighbor in self.get_safe_neighbors(current):
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)
                    
        return []  # 未找到路径

    def plan_next_step(self, current_pos: tuple[int, int], all_possible_targets: list[tuple[int, int]]) -> tuple[int, int]:
        # 优先使用现有路径计划
        if self.path_plan:
            next_step = self.path_plan.popleft()
            if self.is_valid_move(next_step):
                self.last_pos = current_pos
                return next_step
        
        # 检查目标是否可达
        self.target_visible = False
        reachable_targets = [t for t in all_possible_targets 
                            if self.is_valid_move(t) and 
                            self.known_map[t[1], t[0]] == 0]
        
        if reachable_targets:
            self.target_visible = True
            # 尝试直接直线路径
            for target in reachable_targets:
                # 水平方向移动
                if target[1] == current_pos[1]:
                    step_x = target[0] - current_pos[0]
                    if step_x != 0:
                        next_step = (current_pos[0] + (1 if step_x > 0 else -1), current_pos[1])
                        if self.is_valid_move(next_step):
                            self.last_pos = current_pos
                            return next_step
                
                # 垂直方向移动
                if target[0] == current_pos[0]:
                    step_y = target[1] - current_pos[1]
                    if step_y != 0:
                        next_step = (current_pos[0], current_pos[1] + (1 if step_y > 0 else -1))
                        if self.is_valid_move(next_step):
                            self.last_pos = current_pos
                            return next_step
            
            # 如果直线路径不可用，BFS规划路径到最近目标
            closest_target = min(reachable_targets, 
                                key=lambda t: abs(t[0]-current_pos[0]) + abs(t[1]-current_pos[1]))
            
            # 简单BFS搜索路径
            queue = deque([current_pos])
            visited = {current_pos: None}
            
            while queue:
                current = queue.popleft()
                if current == closest_target:
                    # 重建路径
                    path = []
                    while current != current_pos:
                        path.append(current)
                        current = visited[current]
                    self.path_plan = deque(path[::-1])
                    return self.plan_next_step(current_pos, all_possible_targets)
                
                for neighbor in self.get_safe_neighbors(current):
                    if neighbor not in visited:
                        visited[neighbor] = current
                        queue.append(neighbor)
        
        # 探索模式：查找最近的前沿点
        frontier_path = self.find_nearest_frontier_bfs(current_pos)
        if frontier_path:
            self.path_plan = deque(frontier_path)
            return self.plan_next_step(current_pos, all_possible_targets)
        
        # 直接探索：优先选择未知区域
        safe_moves = self.get_safe_neighbors(current_pos)
        if safe_moves:
            # 优先探索未知区域
            for move in safe_moves:
                if self.known_map[move[1], move[0]] == -1:  # 未知区域
                    self.last_pos = current_pos
                    return move
            
            # 如果所有相邻区域都已知，选择探索最少的区域
            best_move = None
            min_explored = float('inf')
            for move in safe_moves:
                # 计算移动后的探索潜力（未探索邻居数量）
                unexplored_count = sum(1 for dx, dy in self.motions
                                       for nx, ny in [(move[0]+dx, move[1]+dy)]
                                       if (0 <= nx < self.map_shape[1] and
                                           0 <= ny < self.map_shape[0] and
                                           not self.explored_map[ny, nx]))
                
                if unexplored_count > min_explored or best_move is None:
                    best_move = move
                    min_explored = unexplored_count
            
            if best_move:
                self.last_pos = current_pos
                return best_move
        
        # 最后防线：安全移动（不应出现，但确保安全）
        for dx, dy in self.motions:
            next_step = (current_pos[0] + dx, current_pos[1] + dy)
            if self.is_valid_move(next_step):
                self.last_pos = current_pos
                return next_step
        
        # 无法移动（非常罕见）
        return current_pos