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
        self.search_limit = 10000  # 搜索节点限制，防止无限搜索
        
    def update_knowledge(self, current_pos: tuple[int, int], local_view: np.ndarray):
        r = self.view_radius
        cx, cy = current_pos
        h, w = self.map_shape
        
        y_start, y_end = max(0, cy - r), min(h, cy + r + 1)
        x_start, x_end = max(0, cx - r), min(w, cx + r + 1)
        
        # 记录值发生变化的点
        updated_points = set()
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                # 只处理值变化的点
                if self.known_map[y, x] != local_view[y - y_start, x - x_start]:
                    updated_points.add((x, y))
                    self.known_map[y, x] = local_view[y - y_start, x - x_start]
        
        # 扩展检查范围：变化点及其四邻域
        points_to_check = set(updated_points)
        for (x, y) in updated_points:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    points_to_check.add((nx, ny))
        
        # 只更新受影响的边界点
        for pt in points_to_check:
            x, y = pt
            # 清除不再符合条件的边界点
            if pt in self.frontier:
                if self.known_map[y, x] != 0:  # 不再是空地
                    self.frontier.discard(pt)
                else:
                    # 检查是否仍然是边界点
                    is_frontier = False
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h and self.known_map[ny, nx] == -1:
                            is_frontier = True
                            break
                    if not is_frontier:
                        self.frontier.discard(pt)
            
            # 添加新边界点
            if self.known_map[y, x] == 0:  # 当前是空地
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and self.known_map[ny, nx] == -1:
                        self.frontier.add(pt)
                        break
                
        # 重新计算边界点优先级
        cx, cy = current_pos
        self.prioritized_frontier = []
        for point in self.frontier:
            x, y = point
            dist = abs(x - cx) + abs(y - cy)
            heapq.heappush(self.prioritized_frontier, (dist, point))

    def hierarchical_astar(self, current_pos, reachable_targets):
        """分层A*搜索 - 针对大视野半径优化"""
        cx, cy = current_pos
        h, w = self.map_shape
        
        # 层级1：粗粒度搜索（降低分辨率）
        grid_size = max(5, min(20, self.view_radius // 10))  # 动态网格大小
        coarse_map = np.zeros((h // grid_size + 1, w // grid_size + 1), dtype=int)
        
        # 创建粗粒度地图
        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                block = self.known_map[y:y+grid_size, x:x+grid_size]
                # 如果块内包含障碍，则标记为不可通行
                coarse_map[y//grid_size, x//grid_size] = 1 if np.any(block == 1) else 0
        
        # 将当前位置和目标转换到粗粒度坐标
        coarse_pos = (cy // grid_size, cx // grid_size)
        coarse_targets = set()
        for target in reachable_targets:
            tx, ty = target
            coarse_targets.add((ty // grid_size, tx // grid_size))
        
        # 在粗粒度地图上运行A*
        open_set = []
        heapq.heappush(open_set, (0, coarse_pos))
        g_score = {coarse_pos: 0}
        parent = {coarse_pos: None}
        
        found_target = None
        counter = 0
        
        while open_set and counter < self.search_limit:
            counter += 1
            f_val, pos = heapq.heappop(open_set)
            
            if pos in coarse_targets:
                found_target = pos
                break
                
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_y, new_x = pos[0] + dy, pos[1] + dx
                new_pos = (new_y, new_x)
                
                if not (0 <= new_y < coarse_map.shape[0] and 0 <= new_x < coarse_map.shape[1]):
                    continue
                    
                if coarse_map[new_y, new_x] != 0:
                    continue
                    
                new_g = g_score[pos] + 1
                
                if new_pos not in g_score or new_g < g_score[new_pos]:
                    g_score[new_pos] = new_g
                    
                    # 计算到最近目标的曼哈顿距离
                    min_dist = min(abs(new_y - ty) + abs(new_x - tx) for ty, tx in coarse_targets)
                    f_score = new_g + min_dist
                    
                    heapq.heappush(open_set, (f_score, new_pos))
                    parent[new_pos] = pos
        
        if not found_target:
            return None
        
        # 回溯粗粒度路径
        path = []
        current = found_target
        while current != coarse_pos:
            path.append(current)
            current = parent[current]
        path.reverse()
        
        # 转换回原坐标中的航路点
        waypoints = []
        for point in path:
            gy, gx = point
            # 网格中心点作为航路点
            waypoints.append((
                gx * grid_size + grid_size // 2, 
                gy * grid_size + grid_size // 2
            ))
        
        # 层级2：细粒度路径连接
        full_path = []
        current = current_pos
        for wp in waypoints:
            # 如果当前位置就是航路点，跳过
            if current == wp:
                continue
                
            # 运行A*连接两点
            segment_path = self.standard_astar(current, wp, set())
            if not segment_path:
                # 连接失败，提前返回
                return None
                
            full_path.extend(segment_path)
            current = wp
        
        return full_path
    
    def standard_astar(self, start, end, reachable_targets=None):
        """标准A*搜索算法"""
        open_set = []
        heapq.heappush(open_set, (0, start))
        g_score = {start: 0}
        parent = {start: None}
        
        counter = 0
        
        while open_set and counter < self.search_limit:
            counter += 1
            f_val, pos = heapq.heappop(open_set)
            x, y = pos
            
            # 如果指定了目标集合且当前位置在其中，找到目标
            if reachable_targets and pos in reachable_targets:
                # 回溯路径
                path = []
                current = pos
                while current != start:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                return path
                
            # 如果指定了单个目标点
            if end and pos == end:
                # 回溯路径
                path = []
                current = pos
                while current != start:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                return path
                
            # 扩展邻居
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                
                if not (0 <= ny < self.map_shape[0] and 0 <= nx < self.map_shape[1]):
                    continue
                    
                if self.known_map[ny, nx] != 0:
                    continue
                    
                new_g = g_score[pos] + 1
                
                if neighbor not in g_score or new_g < g_score[neighbor]:
                    g_score[neighbor] = new_g
                    
                    # 计算启发值
                    if reachable_targets:
                        # 到最近目标的曼哈顿距离
                        h_score = min(abs(nx - tx) + abs(ny - ty) for tx, ty in reachable_targets)
                    elif end:
                        # 到目标点的曼哈顿距离
                        h_score = abs(nx - end[0]) + abs(ny - end[1])
                    else:
                        # 如果没有目标，使用到最近边界点的距离
                        if self.prioritized_frontier:
                            h_score = self.prioritized_frontier[0][0]
                        else:
                            h_score = 0
                    
                    f_score = new_g + h_score
                    heapq.heappush(open_set, (f_score, neighbor))
                    parent[neighbor] = pos
        
        return None

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
            if self.known_map[ny, nx] == 0 and neighbor in self.frontier:
                return neighbor
        
        # 对于大视野半径(r>20)，使用分层A*算法
        if self.view_radius > 20 and reachable_targets:
            self.current_path = self.hierarchical_astar(current_pos, reachable_targets)
            self.last_target = all_possible_targets
            if self.current_path and len(self.current_path) > 0:
                return self.current_path.pop(0)
        
        # 使用标准A*算法搜索目标
        self.current_path = self.standard_astar(current_pos, None, reachable_targets)
        self.last_target = all_possible_targets
        if self.current_path and len(self.current_path) > 0:
            return self.current_path.pop(0)
        
        # 尝试搜索边界点
        self.current_path = self.standard_astar(current_pos, None, self.frontier)
        if self.current_path and len(self.current_path) > 0:
            return self.current_path.pop(0)
        
        # 尝试使用优先边界点
        if self.prioritized_frontier:
            _, nearest_frontier = heapq.heappop(self.prioritized_frontier)
            if nearest_frontier in self.frontier:
                # 尝试直接路径到最近边界点
                self.current_path = self.standard_astar(current_pos, nearest_frontier)
                if self.current_path and len(self.current_path) > 0:
                    return self.current_path.pop(0)
        
        # 最终回退：随机选择有效移动方向
        random.shuffle(motions)
        for dx, dy in motions:
            nx, ny = cx + dx, cy + dy
            if (0 <= ny < self.map_shape[0] and 
                0 <= nx < self.map_shape[1] and
                self.known_map[ny, nx] in (0, -1)):  # 允许未知区域
                return (nx, ny)
        
        # 最终保底：留在原地
        return current_pos