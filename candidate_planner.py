import numpy as np
import random
from collections import deque
import math

class Planner:
    def __init__(self, map_shape: tuple[int, int], view_radius: int):
        self.known_map = np.full(map_shape, -1, dtype=int)
        self.view_radius = view_radius
        self.path_taken = []
        self.map_shape = map_shape
        self.steps = 0  # 添加步数计数器
        
        # Q-learning 参数
        self.alpha = 0.1  # 学习率
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 0.3  # 探索率
        self.epsilon_decay = 0.999  # 探索率衰减
        
        # Q-table: 使用字典存储
        self.q_table = {}
        
        # 动作定义: 索引到方向映射
        self.action_map = {
            0: (0, 1),   # 上
            1: (0, -1),  # 下
            2: (1, 0),   # 右
            3: (-1, 0)   # 左
        }
        
        # 状态追踪
        self.last_state = None
        self.last_action = None
        self.last_pos = None
    
    def update_knowledge(self, current_pos: tuple[int, int], local_view: np.ndarray):
        r = self.view_radius
        x, y = current_pos
        h, w = self.known_map.shape
        
        y_start, y_end = max(0, y - r), min(h, y + r + 1)
        x_start, x_end = max(0, x - r), min(w, x + r + 1)
        
        self.known_map[y_start:y_end, x_start:x_end] = local_view

    def plan_next_step(self, current_pos: tuple[int, int], all_possible_targets: list[tuple[int, int]]) -> tuple[int, int]:
        self.steps += 1
        if not self.path_taken or self.path_taken[-1] != current_pos:
            self.path_taken.append(current_pos)
        
        # 将目标转换为集合便于高效查找
        target_set = set(all_possible_targets)
        
        # 如果当前位置就是目标，则直接返回
        if current_pos in target_set and self.known_map[current_pos[1], current_pos[0]] == 0:
            return current_pos
        
        # 1. 更新Q值（如果可能）
        if self.last_pos is not None and self.last_action is not None:
            # 计算奖励
            reward = self.calculate_reward(self.last_pos, current_pos, target_set)
            
            # 获取新状态
            new_state = self.get_state(current_pos)
            
            # 获取旧状态的Q值
            old_q = self.q_table.get(self.last_state, {}).get(self.last_action, 0)
            
            # 估计未来最大Q值
            max_future_q = max(self.q_table.get(new_state, {}).values()) if new_state in self.q_table else 0
                
            # Q-learning更新公式
            new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
            
            # 更新Q-table
            if self.last_state not in self.q_table:
                self.q_table[self.last_state] = {}
            self.q_table[self.last_state][self.last_action] = new_q
            
            # 衰减探索率
            self.epsilon *= self.epsilon_decay
        
        # 2. 获取当前状态和可能动作
        current_state = self.get_state(current_pos)
        possible_actions = self.get_possible_actions(current_pos)
        
        # 3. 选择动作
        if not possible_actions:
            return current_pos  # 无有效动作
        
        # ε-贪心策略选择
        if random.random() < self.epsilon:
            next_action = random.choice(list(possible_actions.keys()))
        else:
            # 从Q-table中选择最优动作
            q_values = self.q_table.get(current_state, {})
            if q_values:
                # 找最高Q值的动作
                max_q = max(q_values.values())
                best_actions = [a for a, q in q_values.items() if q == max_q and a in possible_actions]
                
                if best_actions:
                    next_action = random.choice(best_actions)
                else:
                    # 后备：基于启发式选择
                    next_action = self.heuristic_action(possible_actions, current_pos, target_set)
            else:
                # 无Q值可用，使用启发式
                next_action = self.heuristic_action(possible_actions, current_pos, target_set)
        
        # 4. 保存当前状态
        self.last_state = current_state
        self.last_action = next_action
        self.last_pos = current_pos
        
        # 5. 返回移动位置
        return possible_actions[next_action]
    
    def get_state(self, pos: tuple[int, int]) -> tuple:
        """简化状态表示"""
        x, y = pos
        h, w = self.known_map.shape
        
        # 状态1: 位置离散化 (3x3网格)
        grid_x = min(2, x // max(1, w // 3))
        grid_y = min(2, y // max(1, h // 3))
        
        # 状态2: 已知目标方向
        target_dir_x, target_dir_y = 0, 0
        for tx, ty in [(w//2, h//2)]:  # 简化：使用地图中心作为目标代理
            target_dir_x = tx - x
            target_dir_y = ty - y
        
        # 离散化方向
        dir_x = -1 if target_dir_x < 0 else (1 if target_dir_x > 0 else 0)
        dir_y = -1 if target_dir_y < 0 else (1 if target_dir_y > 0 else 0)
        
        return (grid_x, grid_y, dir_x, dir_y)
    
    def get_possible_actions(self, pos: tuple[int, int]) -> dict:
        """获取有效动作及其位置"""
        x, y = pos
        h, w = self.known_map.shape
        actions = {}
        
        for action_idx, (dx, dy) in self.action_map.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                # 允许移动到自由空间或未知区域（避免已知障碍物）
                if self.known_map[ny, nx] in (-1, 0):
                    actions[action_idx] = (nx, ny)
        
        return actions
    
    def calculate_reward(self, last_pos: tuple, current_pos: tuple, targets: set) -> float:
        """计算奖励"""
        # 如果到达目标
        if current_pos in targets:
            return 100.0
        
        # 奖励探索新区域
        reward = 0.0
        if self.known_map[current_pos[1], current_pos[0]] == -1:
            reward += 1.0  # 发现新区域奖励
        
        # 小幅惩罚移动（鼓励效率）
        if last_pos != current_pos:
            reward -= 0.01
        
        # 避免停滞惩罚
        if last_pos == current_pos:
            reward -= 0.1
        
        return reward
    
    def heuristic_action(self, possible_actions: dict, current_pos: tuple, targets: set) -> int:
        """后备启发式策略"""
        # 首先尝试向目标移动
        min_dist = float('inf')
        best_action = None
        
        for action_idx, new_pos in possible_actions.items():
            # 计算到目标的曼哈顿距离
            dist = sum(abs(nx - tx) for tx, ty in targets for nx, ny in [new_pos])
            if dist < min_dist:
                min_dist = dist
                best_action = action_idx
        
        if best_action is not None:
            return best_action
        
        # 后备：边界探索
        for action_idx, new_pos in possible_actions.items():
            nx, ny = new_pos
            # 检查是否有未知邻居
            for dx, dy in self.action_map.values():
                nnx, nny = nx + dx, ny + dy
                if 0 <= nnx < self.map_shape[1] and 0 <= nny < self.map_shape[0]:
                    if self.known_map[nny, nnx] == -1:
                        return action_idx
        
        # 最终后备：随机选择
        return random.choice(list(possible_actions.keys()))