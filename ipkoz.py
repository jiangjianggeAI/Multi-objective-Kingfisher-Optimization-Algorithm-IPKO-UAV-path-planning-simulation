import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import random

# -------------------------------------
# 1. 环境定义
# -------------------------------------
   # 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
class Environment:
    def __init__(self):
        # 地形障碍物 (x, y, 高度) - 视为圆柱体进行碰撞检测
        self.terrain_obstacles = [
            {'pos': np.array([5, 8]), 'height': 5.5, 'radius': 1.0}, # 假设一个半径用于碰撞检测
            {'pos': np.array([10, 15]), 'height': 4, 'radius': 1.0},
            {'pos': np.array([15, 7]), 'height': 4.5, 'radius': 1.0}
        ]
        # 雷达威胁 (中心x, 中心y, 中心z, 半径)
        self.radar_threats = [
            {'center': np.array([5, 5, 11]), 'radius': 2.0},
            {'center': np.array([15, 7, 7]), 'radius': 1.7},
            {'center': np.array([17, 18, 5]), 'radius': 1.8},
            {'center': np.array([8, 17, 5]), 'radius': 2.5}
        ]
        # 其他威胁 (中心x, 中心y, 中心z, 半径)
        self.other_threats = [
            {'center': np.array([4, 11, 8]), 'radius': 2.0},
            {'center': np.array([10, 2, 6]), 'radius': 1.7},
            {'center': np.array([5, 6, 6]), 'radius': 1.2},
            {'center': np.array([10, 8, 8]), 'radius': 1.5},
            {'center': np.array([11, 13, 9]), 'radius': 2.0},
            {'center': np.array([13, 14, 5]), 'radius': 1.5},
            {'center': np.array([18, 13, 6]), 'radius': 1.9}
        ]
        # 无人机之间的最小间隔距离
        self.min_uav_separation = 0.5 # 以公里为单位的示例值

    def is_collision(self, point, path_segment=False):
        """检查单个点或路径段是否与任何障碍物/威胁发生碰撞"""
        # 检查地形(简化的圆柱体检测)
        for obs in self.terrain_obstacles:
            dist_xy = np.linalg.norm(point[:2] - obs['pos'])
            if dist_xy < obs['radius'] and point[2] < obs['height']:
                return True # 与地形碰撞

        # 检查雷达威胁(球形检测)
        for threat in self.radar_threats:
            dist = np.linalg.norm(point - threat['center'])
            if dist < threat['radius']:
                return True # 与雷达碰撞

        # 检查其他威胁(球形检测)
        for threat in self.other_threats:
            dist = np.linalg.norm(point - threat['center'])
            if dist < threat['radius']:
                return True # 与其他威胁碰撞

        # 如果需要，添加路径段碰撞检查(沿段插值点)
        # if path_segment: ...

        return False # 无碰撞

    def check_inter_uav_collision(self, paths):
        """检查无人机路径之间的碰撞(简化版)"""
        num_uavs = len(paths)
        if num_uavs < 2:
            return False

        # 假设路径是相同长度的点列表
        # 这是在对应航点上的简化检查
        # 适当的检查需要检查路径段之间的距离
        path_len = len(paths[0])
        for i in range(path_len):
            for uav1_idx in range(num_uavs):
                for uav2_idx in range(uav1_idx + 1, num_uavs):
                    p1 = paths[uav1_idx][i]
                    p2 = paths[uav2_idx][i]
                    if np.linalg.norm(p1 - p2) < self.min_uav_separation:
                        return True # 无人机间碰撞
        return False

# -------------------------------------
# 2. 路径表示与成本函数
# -------------------------------------
class PathPlanner:
    def __init__(self, environment, uav_params, num_waypoints=20):
        self.env = environment
        self.num_uavs = len(uav_params['starts'])
        self.starts = [np.array(s) for s in uav_params['starts']]
        self.ends = [np.array(e) for e in uav_params['ends']]
        self.num_waypoints = num_waypoints # 中间点数量 + 终点

    def generate_initial_path(self, uav_index):
        """初始生成直线路径"""
        start = self.starts[uav_index]
        end = self.ends[uav_index]
        path = np.linspace(start, end, self.num_waypoints)
        return path

    def calculate_path_length(self, path):
        """计算单个路径的总长度"""
        length = 0
        for i in range(len(path) - 1):
            length += np.linalg.norm(path[i+1] - path[i])
        return length

    def calculate_fitness(self, individual_paths):
        """计算所有无人机路径集的适应度(成本)"""
        total_length = 0
        collision_penalty = 0
        inter_uav_penalty = 0
        penalty_scale = 10000 # 碰撞的高惩罚值

        for i in range(self.num_uavs):
            path = individual_paths[i]
            total_length += self.calculate_path_length(path)
            # 检查每个段/航点的碰撞
            for j in range(1, len(path)): # 跳过起点进行碰撞检查
                # 更稳健: 检查path[j-1]和path[j]之间的段
                if self.env.is_collision(path[j]):
                    collision_penalty += 1

        # 检查无人机间碰撞
        if self.env.check_inter_uav_collision(individual_paths):
             inter_uav_penalty += 1

        fitness = total_length + penalty_scale * (collision_penalty + inter_uav_penalty)
        return fitness

# -------------------------------------
# 3. IPKO算法实现(概念性)
# -------------------------------------
class IPKO_Optimizer:
    def __init__(self, planner, pop_size=50, max_iter=100):
        self.planner = planner
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.map_compass_factor = 0.2 # 示例
        self.landmark_factor = 0.5 # 示例

        # 初始化种群: 每只"鸽子"保存所有无人机的路径
        self.population = [] # 路径列表的列表 [[uav1_path, uav2_path,...], ...]
        self.velocities = [] # 对应速度
        self.fitness = []   # 每只鸽子的适应度
        self.pbest_paths = [] # 每只鸽子的个人最佳路径
        self.pbest_fitness = [] # 个人最佳适应度
        self.gbest_paths = None # 迄今为止找到的全局最佳路径
        self.gbest_fitness = float('inf')

    def initialize_population(self):
        self.population = []
        self.velocities = []
        self.fitness = []
        self.pbest_paths = []
        self.pbest_fitness = [float('inf')] * self.pop_size
        self.gbest_paths = None
        self.gbest_fitness = float('inf')

        for _ in range(self.pop_size):
            pigeon_paths = []
            pigeon_velocities = []
            for i in range(self.planner.num_uavs):
                # 生成初始路径(例如直线+小的随机扰动)
                initial_path = self.planner.generate_initial_path(i)
                # 如果需要，为初始路径添加噪声/随机性
                noise = (np.random.rand(*initial_path.shape) - 0.5) * 0.5
                perturbed_path = initial_path + noise
                perturbed_path[0] = self.planner.starts[i] # 确保起点固定
                perturbed_path[-1] = self.planner.ends[i]  # 确保终点固定
                pigeon_paths.append(perturbed_path)

                # 初始速度(可以是零或小的随机值)
                pigeon_velocities.append(np.zeros_like(perturbed_path))

            self.population.append(pigeon_paths)
            self.velocities.append(pigeon_velocities)
            self.pbest_paths.append(list(pigeon_paths)) # 初始pbest是起始路径

            # 计算初始适应度
            fit = self.planner.calculate_fitness(pigeon_paths)
            self.fitness.append(fit)
            self.pbest_fitness[_] = fit

            # 更新全局最佳
            if fit < self.gbest_fitness:
                self.gbest_fitness = fit
                # 需要深拷贝吗？是的，避免在种群变化时修改gbest
                self.gbest_paths = [p.copy() for p in pigeon_paths]

        print(f"初始最佳适应度: {self.gbest_fitness}")


    def run(self):
        self.initialize_population()
        convergence_curve = []

        for t in range(self.max_iter):
            # --- 地图和指南针操作 ---
            # 计算中心(所有无人机所有航点的平均位置)
            # 简单方法: 基于全局最佳和随机因子更新
            center_of_population = self.calculate_population_center() # 需要实现

            for i in range(self.pop_size):
                for uav_idx in range(self.planner.num_uavs):
                    # 更新速度(简化的PSO类更新)
                    r1 = random.random()
                    # 速度更新需要为IPKO特定内容进行细化
                    new_velocity = (self.velocities[i][uav_idx] * 0.9 # 惯性权重
                                    + self.map_compass_factor * r1 * (self.gbest_paths[uav_idx] - self.population[i][uav_idx]))
                                    # 缺少PKO特定项如中心影响

                    # 更新位置(路径航点)
                    new_path = self.population[i][uav_idx] + new_velocity

                    # --- 边界和约束处理 ---
                    # 确保起点/终点保持固定
                    new_path[0] = self.planner.starts[uav_idx]
                    new_path[-1] = self.planner.ends[uav_idx]
                    # 如果需要，将坐标限制在边界内
                    # 可以在此处添加路径平滑或碰撞修复机制

                    # 更新鸽子的状态
                    self.velocities[i][uav_idx] = new_velocity
                    self.population[i][uav_idx] = new_path

            # --- 地标操作 ---
            # 1. 在地图/指南针更新后计算所有鸽子的适应度
            for i in range(self.pop_size):
                self.fitness[i] = self.planner.calculate_fitness(self.population[i])
                 # 更新pbest
                if self.fitness[i] < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = self.fitness[i]
                    self.pbest_paths[i] = [p.copy() for p in self.population[i]] # 存储副本

            # 2. 按适应度排序种群
            sorted_indices = np.argsort(self.fitness)
            sorted_population = [self.population[k] for k in sorted_indices]
            sorted_velocities = [self.velocities[k] for k in sorted_indices]
            sorted_fitness = [self.fitness[k] for k in sorted_indices]
            sorted_pbest_paths = [self.pbest_paths[k] for k in sorted_indices]
            sorted_pbest_fitness = [self.pbest_fitness[k] for k in sorted_indices]


            # 3. 从排序列表中更新全局最佳
            if sorted_fitness[0] < self.gbest_fitness:
                self.gbest_fitness = sorted_fitness[0]
                self.gbest_paths = [p.copy() for p in sorted_population[0]]

            # 4. 丢弃一半(或应用地标逻辑)
            num_to_keep = self.pop_size // 2
            next_gen_population = [None] * self.pop_size
            next_gen_velocities = [None] * self.pop_size
            next_gen_fitness = [float('inf')] * self.pop_size
            next_gen_pbest_paths = [None] * self.pop_size
            next_gen_pbest_fitness = [float('inf')] * self.pop_size


            # 保留最好的一半
            for i in range(num_to_keep):
                next_gen_population[i] = sorted_population[i]
                next_gen_velocities[i] = sorted_velocities[i]
                next_gen_fitness[i] = sorted_fitness[i]
                next_gen_pbest_paths[i] = sorted_pbest_paths[i]
                next_gen_pbest_fitness[i] = sorted_pbest_fitness[i]


            # 5. 基于"目标"(最佳鸽子)更新剩余部分
            # 这里需要确切的IPKO地标更新规则。
            # 简化: 向保留的一半中的最佳移动。
            target_center = self.calculate_target_center(sorted_population[:num_to_keep]) # 需要实现

            for i in range(num_to_keep): # 基于目标更新保留的一半
                 for uav_idx in range(self.planner.num_uavs):
                     # 示例更新 - 需要IPKO特定公式
                     r2 = random.random()
                     # 向目标更新位置(例如，最佳的平均)
                     update_direction = target_center[uav_idx] - next_gen_population[i][uav_idx]
                     next_gen_population[i][uav_idx] += self.landmark_factor * r2 * update_direction
                     # 再次确保起点/终点固定
                     next_gen_population[i][uav_idx][0] = self.planner.starts[uav_idx]
                     next_gen_population[i][uav_idx][-1] = self.planner.ends[uav_idx]


            # 填充另一半(例如，通过变异、交叉或重新初始化 - IPKO特定内容？)
            # 简单方法: 只保留最好的一半并继续(有效地随时间减少种群大小？)
            # 或者: 围绕最佳解决方案随机重新初始化较差的一半？
            # 目前假设我们只继续使用最好的一半(调整pop_size或正确处理)
            # 更好的方法: 基于好的解决方案生成新的解决方案。
            # 我们只是复制最好的来填充剩余槽位(不是理想的PKO/IPKO)
            for i in range(num_to_keep, self.pop_size):
                idx_to_copy = i % num_to_keep # 循环遍历最佳
                next_gen_population[i] = [p.copy() for p in sorted_population[idx_to_copy]]
                next_gen_velocities[i] = [v.copy() for v in sorted_velocities[idx_to_copy]]
                 # 为这些复制/新的重新评估适应度？还是继承？让我们重新评估。
                next_gen_fitness[i] = self.planner.calculate_fitness(next_gen_population[i])
                next_gen_pbest_paths[i] = [p.copy() for p in next_gen_population[i]] # 为新槽位重置pbest
                next_gen_pbest_fitness[i] = next_gen_fitness[i]


            # 为下一次迭代更新种群
            self.population = next_gen_population
            self.velocities = next_gen_velocities
            self.fitness = next_gen_fitness
            self.pbest_paths = next_gen_pbest_paths
            self.pbest_fitness = next_gen_pbest_fitness


            convergence_curve.append(self.gbest_fitness)
            if t % 10 == 0:
                print(f"迭代 {t+1}/{self.max_iter}, 最佳适应度: {self.gbest_fitness}")

        print(f"优化完成。最终最佳适应度: {self.gbest_fitness}")
        return self.gbest_paths, convergence_curve

    def calculate_population_center(self):
        # 需要为路径定义特定定义 - 例如，平均路径
        # 占位符: 返回全局最佳路径作为中心
        if self.gbest_paths:
            return self.gbest_paths
        else: # 初始化后不应发生
            return self.population[0]

    def calculate_target_center(self, best_pigeons):
         # 需要特定定义 - 例如，最佳鸽子的平均路径
         # 占位符: 返回最佳鸽子的路径
        if best_pigeons:
            return best_pigeons[0]
        else:
            return self.population[0]


# -------------------------------------
# 4. 可视化
# -------------------------------------
def plot_results(environment, paths, starts, ends):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制障碍物
    # 地形(简化为带有高度标签的点或低圆柱体)
    for obs in environment.terrain_obstacles:
        ax.scatter(obs['pos'][0], obs['pos'][1], 0, c='brown', marker='^', s=100, label='地形基础')
        ax.text(obs['pos'][0], obs['pos'][1], 0.1, f"H={obs['height']}", color='black')
        # 绘制圆柱体轮廓(可选)
        z = np.linspace(0, obs['height'], 5)
        theta = np.linspace(0, 2 * np.pi, 20)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = obs['pos'][0] + obs['radius'] * np.cos(theta_grid)
        y_grid = obs['pos'][1] + obs['radius'] * np.sin(theta_grid)
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, color='brown')


    # 威胁区域(球体)
    def plot_sphere(center, radius, color, label_prefix):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        ax.plot_surface(x, y, z, color=color, alpha=0.2, label=f'{label_prefix} 威胁区域')

    for i, threat in enumerate(environment.radar_threats):
        plot_sphere(threat['center'], threat['radius'], 'red', '雷达')
    for i, threat in enumerate(environment.other_threats):
         plot_sphere(threat['center'], threat['radius'], 'orange', '其他')

    # 绘制路径
    colors = ['blue', 'green', 'purple']
    for i, path in enumerate(paths):
        path_np = np.array(path)
        ax.plot(path_np[:, 0], path_np[:, 1], path_np[:, 2], marker='.', color=colors[i % len(colors)], label=f'无人机 {i+1} 路径')
        # 起点(圆形标记)
        ax.scatter(starts[i][0], starts[i][1], starts[i][2], c=colors[i % len(colors)], marker='o', s=150, label=f'无人机 {i+1} 起点')
        # 终点(星形标记)
        ax.scatter(ends[i][0], ends[i][1], ends[i][2], c=colors[i % len(colors)], marker='*', s=200, label=f'无人机 {i+1} 终点')
    # 设置坐标轴标签和范围(根据坐标调整)
    ax.set_xlabel("X (公里)")
    ax.set_ylabel("Y (公里)")
    ax.set_zlabel("Z (公里)")
    ax.set_title("IPKO 3D UAV Path Planning Simulation")

    # Create unique labels for legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    # ax.legend(unique_labels.values(), unique_labels.keys())
    # Adjust view angles to mimic Fig 16-18 (Front, Rear, Top)
    # ax.view_init(elev=30., azim=-60) # Default-ish
    # ax.view_init(elev=0, azim=0) # Front view (Y-Z plane focus)
    # ax.view_init(elev=0, azim=180) # Rear view
    ax.view_init(elev=90, azim=-90) # Top view (X-Y plane focus) - like Fig 18

    plt.show()
# -------------------------------------
# 5. 主程序执行
# -------------------------------------
if __name__ == "__main__":
    # 定义无人机起始/结束点
    uav_params = {
        'starts': [(1.5, 1.5, 3), (0, 1.5, 5), (1.5, 0, 4)],
        'ends': [(20, 20, 7), (16, 20, 8), (20, 16, 8)]
    }

    # 创建环境
    env = Environment()
    # 创建路径规划器
    planner = PathPlanner(env, uav_params, num_waypoints=30) # 更多航点=更平滑路径但维度更高
    # 创建优化器
    optimizer = IPKO_Optimizer(planner, pop_size=60, max_iter=200) # 可调整种群规模和最大迭代次数
    # 运行优化
    start_time = time.time()
    best_paths, convergence = optimizer.run()
    end_time = time.time()
    print(f"\n优化耗时 {end_time - start_time:.2f} 秒。")
    # 打印结果分析(类似于论文中的表2和表3)
    print("\n--- 仿真结果 ---")
    total_length = 0
    for i, path in enumerate(best_paths):
        length = planner.calculate_path_length(path)
        print(f"无人机 {i+1} 路径长度: {length:.2f} 公里")
        total_length += length
    print(f"集群总路径长度: {total_length:.2f} 公里")
    # 对最优路径进行最终碰撞检查(适应度已包含惩罚项)
    final_fitness = planner.calculate_fitness(best_paths)
    print(f"最终路径适应度(包含惩罚项): {final_fitness:.2f}")
    # 如需更详细的碰撞报告可在此添加
    # 绘制收敛曲线
    plt.figure()
    plt.plot(convergence)
    plt.title("IPKO 收敛曲线")
    plt.xlabel("迭代次数")
    plt.ylabel("最佳适应度")
    plt.grid(True)
    plt.show()
    # 绘制3D路径图
    plot_results(env, best_paths, planner.starts, planner.ends)