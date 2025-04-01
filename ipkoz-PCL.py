import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import random
import math
import itertools # 用于网格单元迭代

# -------------------------------------
# 1. 环境定义（带空间网格 - 受PCL启发）
# -------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
class Environment:
    def __init__(self, grid_bounds=None, grid_resolution=None):
        # 地形障碍物 (x, y, 高度) - 视为圆柱体进行碰撞检测
        self.terrain_obstacles_def = [
            {'pos': np.array([5, 8]), 'height': 5.5, 'radius': 1.0},
            {'pos': np.array([10, 15]), 'height': 4, 'radius': 1.0},
            {'pos': np.array([15, 7]), 'height': 4.5, 'radius': 1.0}
        ]
        # 雷达威胁区 (中心x, 中心y, 中心z, 半径)
        self.radar_threats_def = [
            {'center': np.array([5, 5, 11]), 'radius': 2.0},
            {'center': np.array([15, 7, 7]), 'radius': 1.7},
            {'center': np.array([17, 18, 5]), 'radius': 1.8},
            {'center': np.array([8, 17, 5]), 'radius': 2.5}
        ]
        # 其他威胁区 (中心x, 中心y, 中心z, 半径)
        self.other_threats_def = [
            {'center': np.array([4, 11, 8]), 'radius': 2.0},
            {'center': np.array([10, 2, 6]), 'radius': 1.7},
            {'center': np.array([5, 6, 6]), 'radius': 1.2},
            {'center': np.array([10, 8, 8]), 'radius': 1.5},
            {'center': np.array([11, 13, 9]), 'radius': 2.0},
            {'center': np.array([13, 14, 5]), 'radius': 1.5},
            {'center': np.array([18, 13, 6]), 'radius': 1.9}
        ]
        # 无人机之间的最小安全距离
        self.min_uav_separation = 0.5 # 示例值，单位 km

        # 合并所有静态障碍物以便进行网格处理
        self.static_obstacles = []
        for i, obs in enumerate(self.terrain_obstacles_def):
            self.static_obstacles.append({'id': f'T{i}', 'type': 'terrain', **obs})
        for i, obs in enumerate(self.radar_threats_def):
            self.static_obstacles.append({'id': f'R{i}', 'type': 'radar', **obs})
        for i, obs in enumerate(self.other_threats_def):
            self.static_obstacles.append({'id': f'O{i}', 'type': 'other', **obs})

        # --- 空间网格设置 ---
        if grid_bounds is None:
            # 自动计算边界，略大于起点/终点和障碍物范围
            all_points_for_bounds = []
            # 添加障碍物中心/位置
            all_points_for_bounds.extend([p['center'] for p in self.radar_threats_def])
            all_points_for_bounds.extend([p['center'] for p in self.other_threats_def])
            all_points_for_bounds.extend([np.append(p['pos'], 0) for p in self.terrain_obstacles_def])
            all_points_for_bounds.extend([np.append(p['pos'], p['height']) for p in self.terrain_obstacles_def])
            # 根据半径添加范围以获得更好的边界框
            for p in self.radar_threats_def + self.other_threats_def:
                 all_points_for_bounds.append(p['center'] + p['radius'])
                 all_points_for_bounds.append(p['center'] - p['radius'])
            for p in self.terrain_obstacles_def:
                 all_points_for_bounds.append(np.array([p['pos'][0]+p['radius'], p['pos'][1]+p['radius'], p['height']]))
                 all_points_for_bounds.append(np.array([p['pos'][0]-p['radius'], p['pos'][1]-p['radius'], 0]))

            # 添加起点/终点（确保它们是NumPy数组以便计算）
            # 假设可能需要无人机参数 - 稍后传递或使用默认值
            default_starts = np.array([(1.5, 1.5, 3), (0, 1.5, 5), (1.5, 0, 4)])
            default_ends = np.array([(20, 20, 7), (16, 20, 8), (20, 16, 8)])
            all_points_for_bounds.extend(default_starts)
            all_points_for_bounds.extend(default_ends)

            if not all_points_for_bounds: # 处理未定义障碍物/点的情况
                 min_coords = np.array([-5, -5, -1])
                 max_coords = np.array([25, 25, 15])
            else:
                all_points_np = np.array(all_points_for_bounds)
                min_coords = np.min(all_points_np, axis=0) - 2 # 添加缓冲区
                max_coords = np.max(all_points_np, axis=0) + 2 # 添加缓冲区

            self.grid_bounds = {'min': min_coords, 'max': max_coords}
            print(f"自动计算的网格边界: Min={self.grid_bounds['min']}, Max={self.grid_bounds['max']}")
        else:
            self.grid_bounds = grid_bounds

        if grid_resolution is None:
            # 定义网格分辨率（每个维度上的单元格数量）
            self.grid_resolution = np.array([10, 10, 5]) # 示例：10x10x5 网格
        else:
            self.grid_resolution = np.array(grid_resolution)

        self.grid_size = self.grid_bounds['max'] - self.grid_bounds['min']
        # 防止网格尺寸在任何维度上为零时出现除零错误
        self.cell_size = np.divide(self.grid_size, self.grid_resolution,
                                  out=np.full_like(self.grid_size, 1e-6), # 防止单元格尺寸为零
                                  where=self.grid_resolution!=0)
        print(f"网格分辨率: {self.grid_resolution}, 单元格尺寸: {self.cell_size}")

        # 初始化网格（列表的列表的列表）
        self.obstacle_grid = [
            [[[] for _ in range(self.grid_resolution[2])] for _ in range(self.grid_resolution[1])]
            for _ in range(self.grid_resolution[0])
        ]
        self._build_grid()
        # --- 空间网格设置结束 ---

    def _get_cell_indices(self, point):
        """计算给定点的网格单元索引 (ix, iy, iz)"""
        if not self.is_within_bounds(point):
            return None # 点在网格外部

        # 计算相对位置并按逆单元格大小缩放
        relative_pos = point - self.grid_bounds['min']
        # 使用 np.divide 进行安全除法
        indices_float = np.divide(relative_pos, self.cell_size,
                                  out=np.zeros_like(relative_pos), # 如果单元格大小为0，则默认为0
                                  where=self.cell_size!=0)

        indices = np.floor(indices_float).astype(int)

        # 将索引严格限制在网格维度内 [0, resolution-1]
        indices = np.maximum(0, np.minimum(indices, self.grid_resolution - 1))
        return tuple(indices)

    def is_within_bounds(self, point):
        """检查一个点是否在定义的网格边界内"""
        # 使用 np.all 进行元素级比较
        return np.all(point >= self.grid_bounds['min']) and np.all(point <= self.grid_bounds['max'])

    def _build_grid(self):
        """用静态障碍物填充空间网格"""
        print("正在构建障碍物的空间网格...")
        obstacles_added_to_grid = 0
        for i, obs in enumerate(self.static_obstacles):
            min_cell_idx = None
            max_cell_idx = None

            # 确定障碍物的边界框（以网格单元表示）
            if obs['type'] == 'terrain':
                # 圆柱体边界框
                center_xy = obs['pos']
                radius = obs['radius']
                height = obs['height']
                min_pt = np.array([center_xy[0] - radius, center_xy[1] - radius, 0])
                max_pt = np.array([center_xy[0] + radius, center_xy[1] + radius, height])
            elif obs['type'] == 'radar' or obs['type'] == 'other':
                # 球体边界框
                center = obs['center']
                radius = obs['radius']
                min_pt = center - radius
                max_pt = center + radius
            else:
                continue # 跳过未知的障碍物类型

            # 在获取单元格索引之前，将边界框裁剪到网格边界
            min_pt_clipped = np.maximum(min_pt, self.grid_bounds['min'])
            max_pt_clipped = np.minimum(max_pt, self.grid_bounds['max'])

            # 获取裁剪后边界框角落的单元格索引
            min_cell_idx = self._get_cell_indices(min_pt_clipped)
            max_cell_idx = self._get_cell_indices(max_pt_clipped)

            # 将障碍物索引添加到其可能重叠的所有单元格中
            if min_cell_idx is not None and max_cell_idx is not None:
                # 确保索引在有效范围内 (0 到 res-1)
                start_ix = max(0, min_cell_idx[0])
                end_ix = min(self.grid_resolution[0] - 1, max_cell_idx[0])
                start_iy = max(0, min_cell_idx[1])
                end_iy = min(self.grid_resolution[1] - 1, max_cell_idx[1])
                start_iz = max(0, min_cell_idx[2])
                end_iz = min(self.grid_resolution[2] - 1, max_cell_idx[2])

                # 遍历有效索引范围内的单元格
                cell_updated = False
                for ix in range(start_ix, end_ix + 1):
                    for iy in range(start_iy, end_iy + 1):
                        for iz in range(start_iz, end_iz + 1):
                             # 检查索引 i 是否已在此单元格的列表中
                             if i not in self.obstacle_grid[ix][iy][iz]:
                                self.obstacle_grid[ix][iy][iz].append(i) # 存储障碍物索引
                                cell_updated = True
                if cell_updated:
                    obstacles_added_to_grid += 1

        print(f"空间网格构建完成。{obstacles_added_to_grid} 个障碍物已添加到网格单元。")


    def is_collision_detail(self, point, obstacle_index):
        """对一个点和一个特定障碍物执行详细的碰撞检查"""
        # 确保 point 是 NumPy 数组
        point = np.asarray(point)
        if point.shape != (3,):
             # 如果需要，处理不正确的点形状，例如返回 False 或引发错误
             # print(f"警告: is_collision_detail 中无效的点形状 {point.shape}")
             return False

        # 检查 obstacle_index 是否有效
        if not (0 <= obstacle_index < len(self.static_obstacles)):
            # print(f"警告: 无效的障碍物索引 {obstacle_index}")
            return False # 无效索引

        obs = self.static_obstacles[obstacle_index]

        if obs['type'] == 'terrain':
            # 确保 obs['pos'] 是 NumPy 数组
            obs_pos = np.asarray(obs['pos'])
            if obs_pos.shape != (2,):
                 # print(f"警告: 无效的地形障碍物位置形状 {obs_pos.shape}")
                 return False # 无效形状

            # 正确计算距离（应为标量）
            dist_xy = np.linalg.norm(point[:2] - obs_pos) # 应为标量
            radius = obs['radius'] # 标量
            height = obs['height'] # 标量

            # 比较前检查距离是否为标量
            if not np.isscalar(dist_xy):
                 # 这种情况表明意外的输入导致了非标量距离
                 # print(f"警告: 为地形碰撞计算了非标量距离: {dist_xy}")
                 return False # 将其视为非碰撞或处理错误

            # *** 已修正的比较 ***
            # 比较标量距离与标量半径
            if dist_xy < radius and point[2] < height:
                return True # 与地形圆柱体碰撞

        elif obs['type'] == 'radar' or obs['type'] == 'other':
            # 确保 obs['center'] 是 NumPy 数组
            center = np.asarray(obs['center'])
            if center.shape != (3,):
                 # print(f"警告: 无效的威胁中心形状 {center.shape}")
                 return False

            # 计算距离（应为标量）
            dist = np.linalg.norm(point - center) # 应为标量
            radius = obs['radius'] # 标量

            if not np.isscalar(dist):
                 # print(f"警告: 为球形威胁计算了非标量距离: {dist}")
                 return False

            # 比较标量距离与标量半径
            if dist < radius:
                return True # 与威胁球体碰撞
        return False

    def is_collision(self, point):
        """使用空间网格检查单个点是否与任何障碍物/威胁发生碰撞"""
        point = np.asarray(point) # 确保点是数组
        # 首先检查点是否在优化区域之外
        if not self.is_within_bounds(point):
            # 点在网格之外。根据需求处理：
            # 1. 假设不安全：返回 True
            # 2. 假设安全（如果边界只覆盖已知障碍物）：返回 False
            # 3. 对所有障碍物执行检查（较慢的备选方案）：
            #    for i in range(len(self.static_obstacles)):
            #        if self.is_collision_detail(point, i): return True
            #    return False
            return False # 选项 2：在此实现中假设网格外部是安全的

        cell_indices = self._get_cell_indices(point)
        if cell_indices is None:
             # 如果点正好在最大边界上（由于浮点精度）或 is_within_bounds 逻辑与 _get_cell_indices 稍有不同，则可能发生这种情况
             # print(f"警告: 点 {point} 在边界内但未能获取单元格索引。")
             return False # 将其视为非碰撞或进一步调查

        ix, iy, iz = cell_indices
        # 验证索引是否在网格维度内（安全检查）
        if not (0 <= ix < self.grid_resolution[0] and
                0 <= iy < self.grid_resolution[1] and
                0 <= iz < self.grid_resolution[2]):
            # print(f"警告: 为点 {point} 计算了无效的单元格索引 {cell_indices}")
            return False # 无效单元格，视为无碰撞

        # 从特定的网格单元获取潜在障碍物
        potential_obstacle_indices = self.obstacle_grid[ix][iy][iz]

        # 仅与此网格单元中列出的障碍物进行检查
        # 使用集合有助于处理障碍物被多次添加到单元格的情况
        checked_indices = set()
        for obs_index in potential_obstacle_indices:
            if obs_index not in checked_indices:
                 if self.is_collision_detail(point, obs_index):
                     return True # 检测到碰撞
                 checked_indices.add(obs_index) # 将此点查询标记为已检查

        return False # 在此单元格中未发现与障碍物的碰撞


    def check_inter_uav_collision(self, paths):
        """检查无人机路径之间的碰撞（简化的航点检查）"""
        num_uavs = len(paths)
        if num_uavs < 2:
            return False

        # 路径结构的基本验证
        if not paths or not isinstance(paths, list): return False
        path_lengths = [len(p) if p is not None else 0 for p in paths]
        if not all(l > 0 for l in path_lengths): return False # 需要至少一个点
        # 确保所有路径具有相同的航点数量以进行此简化检查
        first_path_len = path_lengths[0]
        if not all(l == first_path_len for l in path_lengths):
             # print("警告: 在 check_inter_uav_collision 中路径长度不同。无法执行检查。")
             return False # 无法比较相应的航点

        path_len = first_path_len
        if path_len == 0: return False

        for i in range(path_len): # 检查每个航点索引
            for uav1_idx in range(num_uavs):
                for uav2_idx in range(uav1_idx + 1, num_uavs):
                    # 获取点 - 假设 paths[idx] 是点的列表/数组
                    p1 = np.asarray(paths[uav1_idx][i])
                    p2 = np.asarray(paths[uav2_idx][i])

                    if p1.shape != (3,) or p2.shape != (3,): continue # 跳过格式错误的点

                    # 计算相同航点索引 i 处的点之间的距离
                    if np.linalg.norm(p1 - p2) < self.min_uav_separation:
                        # print(f"在航点 {i} 检测到无人机 {uav1_idx+1} 和 {uav2_idx+1} 之间的碰撞")
                        return True # 检测到无人机间碰撞
        return False

    # 用于获取原始障碍物定义以进行绘图的辅助方法
    def get_terrain_obstacles(self): return self.terrain_obstacles_def
    def get_radar_threats(self): return self.radar_threats_def
    def get_other_threats(self): return self.other_threats_def


# -------------------------------------
# 2. 路径表示与成本函数
# -------------------------------------
class PathPlanner:
    def __init__(self, environment, uav_params, num_waypoints=20):
        self.env = environment
        if not isinstance(uav_params, dict) or 'starts' not in uav_params or 'ends' not in uav_params:
             raise ValueError("uav_params 必须是包含 'starts' 和 'ends' 列表的字典。")

        self.num_uavs = len(uav_params['starts'])
        if self.num_uavs != len(uav_params['ends']):
            raise ValueError("起点数量必须与终点数量匹配。")

        self.starts = [np.array(s) for s in uav_params['starts']]
        self.ends = [np.array(e) for e in uav_params['ends']]
        # 确保 num_waypoints 包含起点和终点，最少为 2
        self.num_waypoints = max(int(num_waypoints), 2)

    def generate_initial_path(self, uav_index):
        """生成初始直线路径"""
        if not (0 <= uav_index < self.num_uavs):
            raise IndexError("uav_index 超出范围。")
        start = self.starts[uav_index]
        end = self.ends[uav_index]
        # Linspace 包含起点和终点
        path = np.linspace(start, end, self.num_waypoints)
        return path

    def calculate_path_length(self, path):
        """计算单条路径的总长度"""
        path = np.asarray(path) # 确保是 NumPy 数组
        if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 3:
            # print(f"警告: calculate_path_length 中无效的路径形状 {path.shape}")
            return 0 # 无效的路径格式
        length = 0.0
        for i in range(len(path) - 1):
            segment_vector = path[i+1] - path[i]
            length += np.linalg.norm(segment_vector)
        return length

    def check_path_collisions(self, path):
        """计算单条路径上的碰撞次数（检查航点）"""
        path = np.asarray(path) # 确保是 NumPy 数组
        if path.ndim != 2 or path.shape[0] < 1 or path.shape[1] != 3:
            return 0 # 无效的路径格式

        collision_count = 0
        # 检查航点（不包括起点 0，除非它是唯一的点）
        start_index_to_check = 1 if len(path) > 1 else 0
        for j in range(start_index_to_check, len(path)):
            if self.env.is_collision(path[j]):
                collision_count += 1
        # 可选：在此处添加分段碰撞检查以提高准确性
        # ...
        return collision_count

    def calculate_fitness(self, individual_paths):
        """计算所有无人机的一组路径的适应度（成本）"""
        total_length = 0.0
        static_collision_penalty = 0
        inter_uav_penalty_flag = False # 标记是否发生任何无人机间碰撞

        # --- 输入验证 ---
        if not isinstance(individual_paths, list) or len(individual_paths) != self.num_uavs:
             # print("警告: calculate_fitness 中无效的 individual_paths 结构")
             return float('inf') # 无效的输入结构

        valid_paths_present = True
        for i in range(self.num_uavs):
            path = individual_paths[i]
            # 检查路径是否为 None、非 NumPy 数组或形状不正确
            if path is None or not isinstance(path, np.ndarray) or \
               path.ndim != 2 or path.shape[0] != self.num_waypoints or path.shape[1] != 3:
                # print(f"警告: calculate_fitness 中无人机 {i} 的路径无效")
                valid_paths_present = False
                break # 一个无效路径使整个解无效

        if not valid_paths_present:
             return float('inf') # 包含无效路径

        # --- 计算 ---
        # 计算每架无人机的长度和静态碰撞
        for i in range(self.num_uavs):
            path = individual_paths[i] # 已在上面验证
            total_length += self.calculate_path_length(path)
            static_collision_penalty += self.check_path_collisions(path)

        # 检查整组路径的无人机间碰撞
        if self.env.check_inter_uav_collision(individual_paths):
             inter_uav_penalty_flag = True

        # --- 适应度计算 ---
        # 使用高惩罚来强烈阻止碰撞
        collision_penalty_cost = 100000.0 # 每个静态碰撞点的成本
        inter_uav_penalty_cost = 200000.0 # 如果发生任何无人机间碰撞，则采用固定成本

        fitness = total_length + \
                  collision_penalty_cost * static_collision_penalty + \
                  (inter_uav_penalty_cost if inter_uav_penalty_flag else 0.0)

        # 确保适应度非负（对于长度+惩罚应该不会发生）
        return max(0.0, fitness)


# -------------------------------------
# 3. IPKO 算法实现（使用 PSO 逻辑的概念性占位符）
# -------------------------------------
class IPKO_Optimizer:
    def __init__(self, planner, pop_size=50, max_iter=100, inertia_weight=0.7, map_compass_factor=1.5, landmark_factor=1.5):
        if not isinstance(planner, PathPlanner):
            raise TypeError("planner 必须是 PathPlanner 的实例")
        self.planner = planner

        # 验证参数
        self.pop_size = max(10, int(pop_size)) # 确保合理的最小种群大小
        self.max_iter = max(1, int(max_iter))
        self.w = float(inertia_weight) # 惯性权重
        self.c1 = float(map_compass_factor) # PBest 学习因子
        self.c2 = float(landmark_factor) # GBest 学习因子
        # 注意：真正的 PKO/IPKO 具有更复杂的操作符/参数。

        # --- 种群初始化 ---
        # population[pigeon_idx][uav_idx] = path (numpy array)
        self.population = [[None] * self.planner.num_uavs for _ in range(self.pop_size)]
        self.velocities = [[None] * self.planner.num_uavs for _ in range(self.pop_size)]
        self.fitness = [float('inf')] * self.pop_size
        # pbest[pigeon_idx] = 此pigeon找到的最佳路径*集*的单个适应度值
        self.pbest_fitness = [float('inf')] * self.pop_size
        # pbest_paths[pigeon_idx][uav_idx] = 相应的最佳路径集
        self.pbest_paths = [[None] * self.planner.num_uavs for _ in range(self.pop_size)]
        # gbest_paths[uav_idx] = 全局找到的 UAV uav_idx 的最佳路径（最佳整体解决方案的一部分）
        self.gbest_paths = [None] * self.planner.num_uavs
        self.gbest_fitness = float('inf') # gbest_paths 集的适应度


    def initialize_population(self):
        print(f"初始化大小为 {self.pop_size} 的种群...")
        initial_best_fitness = float('inf')

        for i in range(self.pop_size):
            current_pigeon_paths = []
            for uav_idx in range(self.planner.num_uavs):
                # 生成初始路径（直线）
                initial_path = self.planner.generate_initial_path(uav_idx)

                # 仅向中间航点添加随机扰动
                noise_scale = 1.5 # 增加噪声尺度以获得更好的初始探索
                perturbed_path = initial_path.copy()
                if self.planner.num_waypoints > 2:
                    noise = (np.random.rand(self.planner.num_waypoints - 2, 3) - 0.5) * noise_scale
                    perturbed_path[1:-1] += noise # 仅将噪声应用于中间点

                # 存储路径和初始速度
                self.population[i][uav_idx] = perturbed_path
                self.velocities[i][uav_idx] = np.zeros_like(perturbed_path)
                current_pigeon_paths.append(perturbed_path)

            # 计算此完整解决方案（所有无人机路径）的初始适应度
            fit = self.planner.calculate_fitness(current_pigeon_paths)
            self.fitness[i] = fit
            # 初始化此pigeon的 pbest
            self.pbest_paths[i] = [p.copy() for p in current_pigeon_paths]
            self.pbest_fitness[i] = fit

            # 更新初始全局最优
            if fit < initial_best_fitness:
                 initial_best_fitness = fit
                 # 存储与此最佳适应度对应的整个路径集
                 self.gbest_paths = [p.copy() for p in current_pigeon_paths]
                 self.gbest_fitness = fit


        # 验证是否找到了有效的全局最优
        if self.gbest_fitness == float('inf'):
             # 尝试查找可用的最小有限适应度值
             finite_fitness_indices = [idx for idx, f in enumerate(self.fitness) if np.isfinite(f)]
             if finite_fitness_indices:
                 best_initial_idx = min(finite_fitness_indices, key=lambda idx: self.fitness[idx])
                 self.gbest_fitness = self.fitness[best_initial_idx]
                 self.gbest_paths = [p.copy() for p in self.population[best_initial_idx]]
                 print(f"初始最佳适应度 (备选): {self.gbest_fitness:.4f}")
             else:
                  # 这表明存在重大问题 - 可能所有初始路径都严重碰撞。
                  # 检查障碍物定义、起点/终点、惩罚尺度。
                  raise ValueError("无法在种群中初始化任何有效路径（有限适应度）！"
                                   "检查环境设置和初始路径生成。")
        else:
             print(f"初始最佳适应度: {self.gbest_fitness:.4f}")


    def run(self):
        """运行概念性 IPKO 优化循环（使用 PSO 逻辑）"""
        if self.gbest_fitness == float('inf'): # 确保初始化成功运行
            self.initialize_population()

        convergence_curve = [self.gbest_fitness]
        print(f"\n开始优化 ({self.max_iter} 次迭代)...")

        for t in range(self.max_iter):
            # --- 阶段 1：更新速度和位置（地图/罗盘算子 - PSO 风格）---
            for i in range(self.pop_size): # 对于每个pigeon（解决方案）
                # 获取完整的当前解和个体最优解
                current_solution_paths = self.population[i]
                pbest_solution_paths = self.pbest_paths[i]

                # 继续前检查有效性（初始化后应有效）
                if self.fitness[i] == float('inf') or self.pbest_fitness[i] == float('inf'):
                     # 跳过适应度无效的pigeon的更新（理想情况下，在初始化后不应经常发生）
                     # print(f"跳过适应度为无穷大的pigeon {i} 的更新。")
                     continue

                for uav_idx in range(self.planner.num_uavs): # 对于解中的每条无人机路径
                    # 获取此无人机路径更新的组件
                    vel = self.velocities[i][uav_idx]
                    pos = current_solution_paths[uav_idx]
                    pbest_path = pbest_solution_paths[uav_idx]
                    gbest_path = self.gbest_paths[uav_idx] # 此特定无人机的全局最优路径

                    # 生成用于更新随机性的随机数
                    r1 = np.random.rand() # 标量随机数
                    r2 = np.random.rand() # 标量随机数

                    # --- 速度更新（仅针对中间航点）---
                    if self.planner.num_waypoints > 2:
                        vel_intermediate = vel[1:-1]
                        pos_intermediate = pos[1:-1]
                        pbest_intermediate = pbest_path[1:-1]
                        gbest_intermediate = gbest_path[1:-1]

                        new_vel_intermediate = (self.w * vel_intermediate +
                                                self.c1 * r1 * (pbest_intermediate - pos_intermediate) +
                                                self.c2 * r2 * (gbest_intermediate - pos_intermediate))
                        # 原地更新速度数组
                        self.velocities[i][uav_idx][1:-1] = new_vel_intermediate
                    else: # 如果路径只有起点和终点 - 无需速度更新
                         new_vel_intermediate = np.array([]) # 空数组


                    # --- 位置更新（仅针对中间航点）---
                    new_pos = pos.copy() # 从当前路径的副本开始
                    if self.planner.num_waypoints > 2:
                        # 根据*新*速度更新中间位置
                        new_pos[1:-1] = pos[1:-1] + self.velocities[i][uav_idx][1:-1]

                        # --- 边界处理（可选 - 将中间点限制在网格边界内）---
                        min_b = self.planner.env.grid_bounds['min']
                        max_b = self.planner.env.grid_bounds['max']
                        new_pos[1:-1] = np.clip(new_pos[1:-1], min_b, max_b)

                    # 存储更新后的路径（起点/终点保持固定）
                    self.population[i][uav_idx] = new_pos

                # --- 适应度计算和 PBest 更新（在更新完pigeon i 的所有无人机路径之后）---
                current_fitness = self.planner.calculate_fitness(self.population[i])
                self.fitness[i] = current_fitness

                # 如果当前解更好，则更新个体最优
                if current_fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = current_fitness
                    self.pbest_paths[i] = [p.copy() for p in self.population[i]] # 存储更好解的副本

            # --- GBest 更新（在评估完该代所有pigeon之后）---
            # 找到当前种群中适应度最佳的pigeon的索引
            best_current_idx = np.argmin(self.fitness) # 适应度最小值对应的索引

            # 如果该代最佳pigeon优于迄今为止找到的全局最优，则更新全局最优
            if self.fitness[best_current_idx] < self.gbest_fitness:
                self.gbest_fitness = self.fitness[best_current_idx]
                # 更新全局最优解（复制路径集）
                self.gbest_paths = [p.copy() for p in self.population[best_current_idx]]

            # --- 阶段 2：地标算子（概念性 - 简化/跳过）---
            # 在真正的 PKO/IPKO 中，此阶段将涉及：
            # 1. 按适应度对pigeon进行排序。
            # 2. 丢弃较差的一半（或基于阈值）。
            # 3. 根据剩余的优良pigeon计算“目标中心”。
            # 4. 更新剩余pigeon的位置，使其朝向该目标中心。
            # 这种类 PSO 的实现跳过了这个不同的阶段。

            # --- 记录收敛性 ---
            convergence_curve.append(self.gbest_fitness)

            # --- 进度输出 ---
            if (t + 1) % 10 == 0 or t == self.max_iter - 1:
                print(f"迭代 {t+1}/{self.max_iter}, 最佳适应度: {self.gbest_fitness:.4f}")


        print(f"\n优化完成。最终最佳适应度: {self.gbest_fitness:.4f}")
        if self.gbest_paths[0] is None or self.gbest_fitness == float('inf'):
             print("错误：优化未能找到有效的全局最优解。")
             return None, convergence_curve # 返回 None 如果未找到有效解

        # 最终检查：确保 gbest_paths 对应于 gbest_fitness
        final_check_fitness = self.planner.calculate_fitness(self.gbest_paths)
        if not np.isclose(final_check_fitness, self.gbest_fitness):
             print(f"警告：最终 gbest_fitness ({self.gbest_fitness}) 与 gbest_paths 的计算适应度 ({final_check_fitness}) 不匹配。重新计算。")
             self.gbest_fitness = final_check_fitness


        return self.gbest_paths, convergence_curve

# -------------------------------------
# 4. 可视化
# -------------------------------------
def plot_results(environment, paths, starts, ends, title="IPKO 3D 无人机路径规划仿真"):
    """绘制 3D 环境、障碍物和规划路径"""
    if not isinstance(environment, Environment):
         print("错误：用于绘图的环境对象无效。")
         return

    fig = plt.figure(figsize=(15, 12)) # 略宽的图形
    ax = fig.add_subplot(111, projection='3d')
    # 设置中文字体（例如 SimHei，如果已安装）
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 或者其他你系统上有的中文字体
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

    # --- 绘制障碍物 ---
    # 地形（圆柱体）
    terrain_obs = environment.get_terrain_obstacles()
    terrain_plotted = False
    if terrain_obs:
        for obs in terrain_obs:
            # 基础标记
            ax.scatter(obs['pos'][0], obs['pos'][1], 0, c='saddlebrown', marker='^', s=120,
                       label='地形基底' if not terrain_plotted else "")
            # 高度标签
            ax.text(obs['pos'][0], obs['pos'][1], 0.1, f"H={obs['height']:.1f}", color='black', fontsize=8, zorder=10)
            # 圆柱体表面
            z_cyl = np.linspace(0, obs['height'], 10)
            theta_cyl = np.linspace(0, 2 * np.pi, 20)
            theta_grid, z_grid = np.meshgrid(theta_cyl, z_cyl)
            x_grid = obs['pos'][0] + obs['radius'] * np.cos(theta_grid)
            y_grid = obs['pos'][1] + obs['radius'] * np.sin(theta_grid)
            ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, color='peru', rstride=5, cstride=5,
                           label='地形障碍物' if not terrain_plotted else "") # 为其中一个添加标签
            terrain_plotted = True

    # 更高效地绘制球体的辅助函数
    def plot_sphere(center, radius, color, ax, label=""):
        u = np.linspace(0, 2 * np.pi, 20) # 减少点数以提高性能
        v = np.linspace(0, np.pi, 20)   # 减少点数以提高性能
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]
        ax.plot_surface(x, y, z, color=color, alpha=0.25, label=label, rstride=4, cstride=4) # 略高的 alpha 透明度

    # 雷达威胁区（球体）
    radar_threats = environment.get_radar_threats()
    radar_plotted = False
    if radar_threats:
        for i, threat in enumerate(radar_threats):
            plot_sphere(threat['center'], threat['radius'], 'red', ax, label='雷达威胁区' if not radar_plotted else "")
            radar_plotted = True

    # 其他威胁区（球体）
    other_threats = environment.get_other_threats()
    other_plotted = False
    if other_threats:
        for i, threat in enumerate(other_threats):
            plot_sphere(threat['center'], threat['radius'], 'darkorange', ax, label='其他威胁区' if not other_plotted else "")
            other_plotted = True

    # --- 绘制路径 ---
    colors = ['mediumblue', 'forestgreen', 'purple'] # 不同的颜色
    paths_exist = isinstance(paths, list) and len(paths) > 0 and paths[0] is not None

    if paths_exist and len(paths) == len(starts) == len(ends):
        for i, path in enumerate(paths):
            path_np = np.asarray(path)
            if path_np.ndim == 2 and path_np.shape[0] >= 2 and path_np.shape[1] == 3:
                # 绘制路径线
                ax.plot(path_np[:, 0], path_np[:, 1], path_np[:, 2], marker='.', markersize=4, linestyle='-', linewidth=2.0, color=colors[i % len(colors)], label=f'无人机 {i+1} 路径', zorder=5)
                # 清晰地绘制起点/终点
                ax.scatter(starts[i][0], starts[i][1], starts[i][2], c=colors[i % len(colors)], marker='o', s=180, edgecolor='black', label=f'无人机 {i+1} 起点', depthshade=False, zorder=6)
                ax.scatter(ends[i][0], ends[i][1], ends[i][2], c=colors[i % len(colors)], marker='*', s=300, edgecolor='black', label=f'无人机 {i+1} 终点', depthshade=False, zorder=6)
            else:
                 print(f"警告：无人机 {i+1} 的路径格式无效，无法绘制。")
                 # 如果路径无效，可选地只绘制起点/终点
                 ax.scatter(starts[i][0], starts[i][1], starts[i][2], c=colors[i % len(colors)], marker='o', s=180, edgecolor='black', label=f'无人机 {i+1} 起点 (无路径)', depthshade=False, zorder=6)
                 ax.scatter(ends[i][0], ends[i][1], ends[i][2], c=colors[i % len(colors)], marker='*', s=300, edgecolor='black', label=f'无人机 {i+1} 终点 (无路径)', depthshade=False, zorder=6)
    else:
         print("警告：未提供有效的绘图路径或路径/起点/终点数量不匹配。")
         # 如果没有可用路径，则仅绘制起点/终点
         if starts and ends and len(starts) == len(ends):
             for i in range(len(starts)):
                 ax.scatter(starts[i][0], starts[i][1], starts[i][2], c=colors[i % len(colors)], marker='o', s=180, edgecolor='black', label=f'无人机 {i+1} 起点', depthshade=False, zorder=6)
                 ax.scatter(ends[i][0], ends[i][1], ends[i][2], c=colors[i % len(colors)], marker='*', s=300, edgecolor='black', label=f'无人机 {i+1} 终点', depthshade=False, zorder=6)

    # --- 坐标轴和标签 ---
    min_b = environment.grid_bounds['min']
    max_b = environment.grid_bounds['max']
    ax.set_xlim(min_b[0], max_b[0])
    ax.set_ylim(min_b[1], max_b[1])
    ax.set_zlim(min_b[2], max_b[2]) # 使用网格边界作为 Z 轴限制

    ax.set_xlabel("X (km)", fontsize=10)
    ax.set_ylabel("Y (km)", fontsize=10)
    ax.set_zlabel("Z (km)", fontsize=10)
    ax.set_title(title, fontsize=14)

    # --- 图例处理 ---
    handles, labels = ax.get_legend_handles_labels()
    # 使用字典创建唯一标签
    by_label = dict(zip(labels, handles))
    # 定义首选顺序
    order = [f'无人机 {i+1} 起点' for i in range(self.planner.num_uavs)] + \
            [f'无人机 {i+1} 路径' for i in range(self.planner.num_uavs)] + \
            [f'无人机 {i+1} 终点' for i in range(self.planner.num_uavs)] + \
            ['地形基底', '地形障碍物', '雷达威胁区', '其他威胁区']
    # 基于首选顺序创建排序列表，将已知项放在前面
    sorted_handles = [by_label[lab] for lab in order if lab in by_label]
    sorted_labels = [lab for lab in order if lab in by_label]
    # 添加任何不在首选顺序中的剩余标签
    remaining_labels = [lab for lab in by_label if lab not in order]
    sorted_handles.extend([by_label[lab] for lab in remaining_labels])
    sorted_labels.extend(remaining_labels)

    # 将图例放置在绘图区域之外
    ax.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=9)

    # --- 调整布局并显示绘图 ---
    plt.tight_layout(rect=[0, 0, 0.88, 1]) # 调整右边距以为图例留出空间

    # 设置视图并显示的函数
    def show_view(elevation, azimuth, view_title):
        fig.suptitle(view_title, fontsize=16) # 使用 suptitle 作为整个图形的标题
        ax.view_init(elev=elevation, azim=azimuth)
        plt.draw() # 更新绘图视图
        plt.show(block=True) # block=True 等待窗口关闭

    print("\n显示绘图...")
    show_view(elev=90, azim=-90, view_title="顶视图（等效于图 18）")
    show_view(elev=0, azim=-90, view_title="前视图（等效于图 16 - 沿 Y 轴视图）")
    show_view(elev=0, azim=0, view_title="侧视图（沿 X 轴视图）")
    show_view(elev=25, azim=-120, view_title="等轴测视图")


def plot_convergence(convergence_curve):
    """绘制迭代过程中的适应度收敛曲线"""
    if not convergence_curve or not isinstance(convergence_curve, list):
         print("警告：无收敛数据可供绘制。")
         return

    plt.figure(figsize=(9, 5)) # 调整后的尺寸
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

    # 如果存在，则过滤掉初始的 inf 值，但跟踪起始迭代
    valid_data = [(i, f) for i, f in enumerate(convergence_curve) if np.isfinite(f)]
    if not valid_data:
         print("警告：收敛数据不包含有限值。")
         return
    iterations, fitness_values = zip(*valid_data)

    plt.plot(iterations, fitness_values, linewidth=2, color='teal')
    plt.title("IPKO 收敛曲线 (每次迭代的最佳适应度)", fontsize=12)
    plt.xlabel("迭代次数", fontsize=10)
    plt.ylabel("最佳适应度 (成本)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    # 如果值非常大或小，则使用科学计数法
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.show(block=True)


# -------------------------------------
# 5. 主执行块
# -------------------------------------
if __name__ == "__main__":
    # --- 配置 ---
    uav_params = {
        'starts': [(1.5, 1.5, 3), (0, 1.5, 5), (1.5, 0, 4)],
        'ends': [(20, 20, 7), (16, 20, 8), (20, 16, 8)]
    }
    NUM_WAYPOINTS = 35     # 每条路径中的点数（包括起点/终点）
    POP_SIZE = 60          # 种群大小（候选解的数量）
    MAX_ITER = 200         # 最大优化迭代次数

    # 类 PSO 的概念性参数（需要调整）
    INERTIA = 0.729        # 惯性权重（平衡探索/利用）
    MAP_COMPASS_FACTOR = 1.49 # 个体最优影响（认知部分）
    LANDMARK_FACTOR = 1.49   # 全局最优影响（社会部分）

    # 环境网格参数
    # 更高分辨率 = 更小单元格 -> 如果稀疏则查询可能更快，但需要更多内存/构建时间
    GRID_RESOLUTION = [15, 15, 10] # X, Y, Z 维度的单元格数量

    # --- 设置 ---
    start_setup_time = time.time()
    print("--- 设置环境 ---")
    try:
        env = Environment(grid_resolution=GRID_RESOLUTION)
    except Exception as e:
        print(f"环境设置期间出错: {e}")
        exit()

    print("\n--- 设置路径规划器 ---")
    try:
        planner = PathPlanner(env, uav_params, num_waypoints=NUM_WAYPOINTS)
    except Exception as e:
        print(f"PathPlanner 设置期间出错: {e}")
        exit()

    print("\n--- 设置 IPKO 优化器 ---")
    try:
        optimizer = IPKO_Optimizer(planner,
                                   pop_size=POP_SIZE,
                                   max_iter=MAX_ITER,
                                   inertia_weight=INERTIA,
                                   map_compass_factor=MAP_COMPASS_FACTOR,
                                   landmark_factor=LANDMARK_FACTOR)
    except Exception as e:
        print(f"IPKO_Optimizer 设置期间出错: {e}")
        exit()
    end_setup_time = time.time()
    print(f"\n设置在 {end_setup_time - start_setup_time:.2f} 秒内完成。")

    # --- 运行优化 ---
    print("\n--- 开始 IPKO 优化 ---")
    start_opt_time = time.time()
    try:
        best_paths, convergence = optimizer.run()
    except Exception as e:
        print(f"\n优化运行期间出错: {e}")
        best_paths, convergence = None, [] # 确保变量在出错时也存在
    end_opt_time = time.time()

    if best_paths is not None:
         print(f"\n优化在 {end_opt_time - start_opt_time:.2f} 秒内完成。")
    else:
         print(f"\n优化在 {end_opt_time - start_opt_time:.2f} 秒后失败。")

    # --- 结果分析 ---
    if best_paths and all(p is not None for p in best_paths):
        print("\n--- 模拟结果分析 ---")
        total_length = 0.0
        collision_counts = {'terrain': 0, 'radar': 0, 'other': 0}
        final_static_collisions_total = 0

        for i, path in enumerate(best_paths):
            length = planner.calculate_path_length(path)
            print(f"无人机 {i+1} 路径长度: {length:.3f} km")
            total_length += length

            # 最终路径的详细碰撞检查（检查航点）
            path_waypoint_collisions = 0
            for point_idx, point in enumerate(path):
                 # 可选地跳过起点的碰撞检查？这里检查所有点。
                 # if point_idx == 0: continue
                 if env.is_collision(point):
                     path_waypoint_collisions += 1
                     # 查找导致碰撞的障碍物类型（用于统计）
                     cell_indices = env._get_cell_indices(point)
                     collision_found_type = False
                     if cell_indices:
                         potential_indices = env.obstacle_grid[cell_indices[0]][cell_indices[1]][cell_indices[2]]
                         for obs_idx in potential_indices:
                             if env.is_collision_detail(point, obs_idx):
                                 obs_type = env.static_obstacles[obs_idx]['type']
                                 collision_counts[obs_type] += 1
                                 collision_found_type = True
                                 break # 计算此点找到的第一个碰撞类型
                     # 如果点在网格外部导致碰撞（如果 is_collision 处理了这种情况）
                     if not collision_found_type and env.is_collision(point):
                          # 回退检查（如果 is_collision 检查网格外部）
                          for obs_idx in range(len(env.static_obstacles)):
                              if env.is_collision_detail(point, obs_idx):
                                  obs_type = env.static_obstacles[obs_idx]['type']
                                  collision_counts[obs_type] += 1
                                  break

            print(f"  - 航点静态碰撞次数: {path_waypoint_collisions}")
            final_static_collisions_total += path_waypoint_collisions

        print(f"\n总群体路径长度: {total_length:.3f} km")
        final_inter_uav_collision = env.check_inter_uav_collision(best_paths)
        print(f"最终静态碰撞 (总航点数): {final_static_collisions_total}")
        print(f"  - 地形碰撞次数: {collision_counts['terrain']}")
        print(f"  - 雷达碰撞次数:   {collision_counts['radar']}")
        print(f"  - 其他碰撞次数:   {collision_counts['other']}")
        print(f"检测到最终无人机间碰撞 (航点检查): {'是' if final_inter_uav_collision else '否'}")

        # 最终适应度分数验证
        final_fitness = planner.calculate_fitness(best_paths)
        print(f"最终路径集适应度 (成本): {final_fitness:.3f}")
        if not np.isclose(final_fitness, optimizer.gbest_fitness):
             print(f"注意：最终计算的适应度与优化器的 gbest ({optimizer.gbest_fitness:.3f}) 略有不同 - 可能是浮点精度问题。")

        # --- 绘图 ---
        try:
            plot_convergence(convergence)
            # 修正 plot_results 调用，确保传递正确的参数数量
            plot_results(env, best_paths, planner.starts, planner.ends, title="IPKO 3D 无人机路径规划仿真")
        except Exception as e:
            print(f"\n绘图期间出错: {e}")

    else:
        print("\n--- 优化未能产生有效的最终路径集。 ---")
        # 可选地，无论如何都绘制环境
        try:
             print("绘制没有最终路径的环境设置。")
             plot_results(env, None, planner.starts, planner.ends, title="IPKO 设置 - 未找到有效路径")
        except Exception as e:
            print(f"\n环境绘图期间出错: {e}")

    print("\n--- 模拟结束 ---")