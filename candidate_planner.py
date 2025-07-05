import heapq
import numpy as np
from collections import deque
from scipy.ndimage import maximum_filter, label
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

class Planner:
    def __init__(self, map_shape: tuple[int, int], view_radius: int):
        self.known_map = np.full(map_shape, -1, dtype=int)  # -1: 未知, 0: 自由空间, 1: 障碍
        self.view_radius = view_radius
        self.map_shape = map_shape
        self.motions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 可能的移动方向
        
        # 探索状态图
        self.frontier_map = np.zeros(map_shape, dtype=bool)  # 前沿点标记
        self.explored_map = np.zeros(map_shape, dtype=bool)  # 已探索区域
        self.frontier_clusters = {}  # 前沿簇信息 {中心点: 簇大小}
        self.path_plan = deque()  # 当前路径计划，队列
        self.last_pos = None  # 上一个位置
        self.target_visible = False  # 目标是否可见
        self.path_taken = []  # 历史路径
        
        # 前沿检测核
        self.frontier_kernel = np.array([[0, 1, 0],
                                        [1, 0, 1],
                                        [0, 1, 0]], dtype=bool)
        # 簇检测核
        self.cluster_kernel = np.ones((3, 3), dtype=bool)
        
        # 用于高效距离计算的数据结构
        self.cluster_tree = None
        self.cluster_points = []
        
        # 视野半径阈值，用于切换探索策略
        self.CLUSTER_THRESHOLD = 20  # 视野半径大于此值使用聚类优化
    
    def update_knowledge(self, current_pos: tuple[int, int], local_view: np.ndarray):
        """使用当前视野更新已知地图和前沿点信息"""
        r = self.view_radius
        x, y = current_pos
        h, w = self.map_shape
        
        # 计算当前视野范围
        y_start, y_end = max(0, y - r), min(h, y + r + 1)
        x_start, x_end = max(0, x - r), min(w, x + r + 1)
        
        # 更新已知地图
        self.known_map[y_start:y_end, x_start:x_end] = local_view
        self.explored_map[y_start:y_end, x_start:x_end] = True
        
        # 计算需要更新的区域（扩大一圈以覆盖可能受影响的节点）
        y_low, y_high = max(0, y_start - 1), min(h, y_end + 1)
        x_low, x_high = max(0, x_start - 1), min(w, x_end + 1)
        
        # 重置受影响区域的前沿标记
        self.frontier_map[y_low:y_high, x_low:x_high] = False
        
        # 处理受影响区域
        region = self.known_map[y_low:y_high, x_low:x_high]
        is_free = (region == 0)
        
        # 创建未知区域掩码
        unknown_mask = (region == -1)
        
        # 使用卷积检测前沿点（至少有一个未知邻居的自由区域）
        conv_result = maximum_filter(unknown_mask, footprint=self.frontier_kernel, mode='constant', cval=0)
        has_unknown_neighbor = conv_result & is_free
        
        # 更新前沿图
        self.frontier_map[y_low:y_high, x_low:x_high] = has_unknown_neighbor
        
        # 只有当视野半径大于阈值时才进行聚类优化
        if r > self.CLUSTER_THRESHOLD and np.any(has_unknown_neighbor):
            # 移除受影响区域内旧的簇信息
            self.remove_clusters_in_area((x_low, y_low), (x_high, y_high))
            
            # 对新检测到的区域进行聚类
            labeled, num_clusters = label(has_unknown_neighbor, structure=self.cluster_kernel)
            
            # ===== 优化部分：使用向量化操作替代循环 =====
            if num_clusters > 0:
                # 获取所有簇的点坐标
                cluster_indices = np.arange(1, num_clusters + 1)
                
                # 一次性计算所有簇的中心点
                clusters = [np.argwhere(labeled == i) for i in cluster_indices]
                
                # 过滤空簇
                clusters = [c for c in clusters if c.size > 0]
                
                if clusters:
                    # 计算每个簇的全局坐标
                    global_coords = [
                        (x_low + c[:, 1], y_low + c[:, 0])  # (x坐标数组, y坐标数组)
                        for c in clusters
                    ]
                    
                    # 计算每个簇的中心点
                    centers = [
                        (int(np.mean(xs)), int(np.mean(ys))) 
                        for xs, ys in global_coords
                    ]
                    
                    # 计算每个簇的大小
                    sizes = [len(c) for c in clusters]
                    
                    # 批量更新簇信息
                    for center, size in zip(centers, sizes):
                        self.frontier_clusters[center] = size
                        self.cluster_points.append(center)
            
            # 更新KD树用于高效距离计算
            if self.cluster_points:
                self.cluster_tree = cKDTree(self.cluster_points)
    
    def remove_clusters_in_area(self, top_left, bottom_right):
        """移除指定区域内的簇信息"""
        x_min, y_min = top_left
        x_max, y_max = bottom_right
        
        # 将簇中心点转换为NumPy数组以便向量化操作
        all_centers = np.array(list(self.frontier_clusters.keys()))
        
        if len(all_centers) == 0:
            return
        
        # 向量化筛选区域内的簇中心
        x_vals = all_centers[:, 0]
        y_vals = all_centers[:, 1]
        
        # 创建布尔掩码标记需要移除的簇
        in_area_mask = (
            (x_vals >= x_min) & 
            (x_vals < x_max) & 
            (y_vals >= y_min) & 
            (y_vals < y_max)
        )
        
        # 获取需要移除的簇中心
        clusters_to_remove = all_centers[in_area_mask]
        
        # 没有需要移除的簇时直接返回
        if len(clusters_to_remove) == 0:
            return
        
        # 批量从字典中移除簇
        for center in clusters_to_remove:
            # 转换为元组（NumPy数组不可哈希）
            center_tuple = tuple(center)
            del self.frontier_clusters[center_tuple]
        
        # 批量从簇点列表中移除
        if self.cluster_points:
            # 转换为NumPy数组进行向量化操作
            cluster_points_arr = np.array(self.cluster_points)
            
            # 创建不在移除区域的掩码
            remove_set = {tuple(c) for c in clusters_to_remove}
            keep_mask = np.array([tuple(p) not in remove_set for p in cluster_points_arr])
            
            # 保留不在移除区域的簇点
            self.cluster_points = [tuple(p) for p in cluster_points_arr[keep_mask]]
    
    def is_valid_move(self, pos):
        """检查位置是否可移动（在地图内且不是障碍）"""
        x, y = pos
        return (0 <= x < self.map_shape[1] and 
                0 <= y < self.map_shape[0] and 
                self.known_map[y, x] != 1)
    
    def get_safe_neighbors(self, current_pos):
        """获取当前位置的安全邻居（可移动位置）"""
        return [(current_pos[0] + dx, current_pos[1] + dy) 
                for dx, dy in self.motions
                if self.is_valid_move((current_pos[0] + dx, current_pos[1] + dy))]

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
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
        return []  # 未找到路径

    def heuristic(self, a: tuple[int, int], b: tuple[int, int]):
        """曼哈顿距离启发函数"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def find_nearest_frontier_bfs(self, start_pos):
        """BFS查找最近的前沿点路径（针对小视野优化）"""
        # 如果起点就是前沿点，返回空路径
        if self.frontier_map[start_pos[1], start_pos[0]]:
            return [start_pos]
        
        # 初始化队列和访问记录
        queue = deque([start_pos])
        visited = np.zeros(self.map_shape, dtype=bool)
        visited[start_pos[1], start_pos[0]] = True
        came_from = {}
        
        while queue:
            current = queue.popleft()
            x, y = current
            
            # 找到前沿点时构建路径
            if self.frontier_map[y, x]:
                path = []
                while current != start_pos:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            
            # 探索邻居
            for neighbor in self.get_safe_neighbors(current):
                nx, ny = neighbor
                if not visited[ny, nx]:
                    visited[ny, nx] = True
                    came_from[neighbor] = current
                    queue.append(neighbor)
                    
        return []  # 未找到前沿点

    def select_best_target(self, current_pos, possible_targets):
        """选择价值最高的可达目标（优先可见目标，多个可见目标时选最近）"""
        if not possible_targets:
            return None
        
        # 情况1：只有一个目标点
        if len(possible_targets) == 1:
            return possible_targets[0]
        
        # 获取所有可见目标点
        visible_targets = [
            t for t in possible_targets
            if (abs(t[0] - current_pos[0]) <= self.view_radius and 
                abs(t[1] - current_pos[1]) <= self.view_radius)
        ]
        
        # 情况2：有可见目标点
        if visible_targets:
            # 如果有多个可见目标，选择距离最小的
            if len(visible_targets) > 1:
                # 计算所有可见目标的距离
                distances = [abs(t[0]-current_pos[0]) + abs(t[1]-current_pos[1]) 
                        for t in visible_targets]
                # 返回最近的目标
                return visible_targets[np.argmin(distances)]
            # 只有一个可见目标
            return visible_targets[0]
        
        # 情况3：没有可见目标，选择所有目标中最近的
        distances = [abs(t[0]-current_pos[0]) + abs(t[1]-current_pos[1]) 
                for t in possible_targets]
        return possible_targets[np.argmin(distances)]
    
    def select_best_frontier_cluster(self, current_pos):
        """选择最优前沿簇（平衡簇大小和距离，大视野优化）"""
        if not self.frontier_clusters or not self.cluster_tree:
            return None
        
        # 使用KD树高效查询最近邻
        current_arr = np.array([current_pos])
        k = min(10, len(self.cluster_points))
        distances, indices = self.cluster_tree.query(current_arr, k=k)#cluster_tree是由KDTree构建的
        
        if distances.size == 0:
            return None
            
        best_score = -float('inf')
        best_cluster = None
        
        # 计算候选簇的综合得分
        for i in range(k):
            if np.isinf(distances[0][i]):
                continue
                
            idx = indices[0][i]
            cluster_center = self.cluster_points[idx]
            cluster_value = self.frontier_clusters[cluster_center]
            distance = distances[0][i]
            
            # 得分 = 簇大小 / (距离 + 0.001)
            score = cluster_value / (distance + 0.001)
            
            if score > best_score:
                best_score = score
                best_cluster = cluster_center
        
        return best_cluster

    def plan_next_step(self, current_pos: tuple[int, int], all_possible_targets: list[tuple[int, int]]) -> tuple[int, int]:
        """计划下一步行动，返回下一步位置"""
        self.path_taken.append(current_pos)
        
        # 层次1: 继续现有路径
        if self.path_plan:
            next_step = self.path_plan.popleft()
            if self.is_valid_move(next_step):
                self.last_pos = current_pos
                return next_step
        
        # 层次2: 目标导向探索
        reachable_targets = [t for t in all_possible_targets 
                           if self.is_valid_move(t) and 
                           self.known_map[t[1], t[0]] != 1]  # 可达的非障碍物目标
        
        if reachable_targets:
            best_target = self.select_best_target(current_pos, reachable_targets)
            if best_target:
                path = self.a_star_search(current_pos, best_target)
                if path:
                    self.path_plan = deque(path)
                    return self.plan_next_step(current_pos, all_possible_targets)
        
        # 层次3: 前沿探索（根据视野切换策略）
        if self.view_radius <= self.CLUSTER_THRESHOLD:
            # 小视野策略：直接BFS寻找最近前沿点
            frontier_path = self.find_nearest_frontier_bfs(current_pos)
            if frontier_path:
                # 从路径中提取下一步位置
                self.path_plan = deque(frontier_path)
                return self.plan_next_step(current_pos, all_possible_targets)
        else:
            # 大视野策略：聚类优化探索
            best_frontier = self.select_best_frontier_cluster(current_pos)
            if best_frontier:
                path = self.a_star_search(current_pos, best_frontier)
                if path:
                    self.path_plan = deque(path)
                    return self.plan_next_step(current_pos, all_possible_targets)
        
        # 层次4: 本地启发探索
        safe_moves = self.get_safe_neighbors(current_pos)
        if safe_moves:
            # 优先选择未知区域
            for move in safe_moves:
                if self.known_map[move[1], move[0]] == -1:
                    self.last_pos = current_pos
                    return move
            
            # 其次选择未探索方向最多的邻居
            best_move = None
            max_unexplored = -1
            
            for move in safe_moves:
                unexplored_count = 0
                for dx, dy in self.motions:
                    nx, ny = move[0] + dx, move[1] + dy
                    if (0 <= nx < self.map_shape[1] and
                        0 <= ny < self.map_shape[0] and
                        not self.explored_map[ny, nx]):
                        unexplored_count += 1
                
                if unexplored_count > max_unexplored:
                    best_move = move
                    max_unexplored = unexplored_count
            
            if best_move:
                self.last_pos = current_pos
                return best_move
        
        # 层次5: 安全回退策略
        for dx, dy in self.motions:
            next_step = (current_pos[0] + dx, current_pos[1] + dy)
            if self.is_valid_move(next_step):
                self.last_pos = current_pos
                return next_step
        
        # 无法移动，停留在原地
        return current_pos