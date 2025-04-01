# 斑翠鸟优化算法（IPKO）项目实现说明

## 一、项目概述
本项目主要实现了基于斑翠鸟优化算法（IPKO）的无人机路径规划。通过模拟斑翠鸟的捕食行为，结合空间网格环境模型，对无人机路径进行优化，以避免障碍物和威胁，并最小化路径成本。

## 二、运行环境
### 1. Python 版本
建议使用 Python 3.7 及以上版本。

### 2. 依赖库
- `numpy`：用于数值计算。
- `matplotlib`：用于数据可视化。
- `mpl_toolkits.mplot3d`：用于绘制 3D 图形。
- `time`：用于计时。
- `random`：用于生成随机数。
- `math`：用于数学计算。
- `itertools`：用于迭代操作。

你可以使用以下命令安装所需的依赖库：
```bash
pip install numpy matplotlib
```

## 三、代码文件说明

### 1. `ipkoz-PCL.py`
#### 功能概述
该文件实现了基于空间网格的环境定义、路径表示与成本函数、IPKO 算法以及可视化功能。主要用于无人机路径规划的整体流程控制。

#### 代码块功能
- **环境定义 (`Environment` 类)**：
  - `__init__`：初始化环境，包括网格边界和分辨率。
  - `_get_cell_indices`：计算给定点所在的网格单元索引。
  - `is_within_bounds`：检查点是否在定义的网格边界内。
  - `_build_grid`：将静态障碍物填充到空间网格中。
  - `is_collision_detail`：对给定点与特定障碍物进行详细的碰撞检查。
  - `is_collision`：检查单个点是否与任何障碍物或威胁发生碰撞。
  - `check_inter_uav_collision`：检查无人机路径之间的碰撞。
  - `get_terrain_obstacles`、`get_radar_threats`、`get_other_threats`：获取不同类型的障碍物定义用于绘图。

- **路径规划 (`PathPlanner` 类)**：
  - `__init__`：初始化路径规划器，包括环境和无人机参数。
  - `generate_initial_path`：初始生成直线路径。
  - `calculate_path_length`：计算单个路径的总长度。
  - `check_path_collisions`：统计单个路径上的碰撞次数。
  - `calculate_fitness`：计算所有无人机路径集的适应度（成本）。

- **IPKO 算法实现 (`IPKO_Optimizer` 类)**：
  - `__init__`：初始化优化器，包括路径规划器、种群大小、最大迭代次数等参数。
  - `initialize_population`：初始化种群。
  - `run`：运行 IPKO 优化循环。

- **可视化函数**：
  - `plot_results`：绘制 3D 环境、障碍物和规划路径。
  - `plot_convergence`：绘制适应度收敛曲线。

### 2. `pko.py`
#### 功能概述
该文件实现了斑翠鸟优化算法（PKO）的基本版本，包括 Kent 映射、种群初始化、适应度函数计算和优化过程。

#### 代码块功能
- **Kent 映射 (`kent_map`)**：生成 Kent 混沌序列。
- **种群初始化 (`initialize_population`)**：使用 Kent 映射生成初始种群。
- **适应度函数 (`fitness_function`)**：计算路径的适应度值。
- **α - 稳定分布 (`alpha_stable_distribution`)**：简化为高斯分布。
- **IPKO 算法主函数 (`ipko`)**：实现 IPKO 优化过程，包括动态聚类、栖息与悬停、潜水策略和共生阶段。

### 3. `ipko.py`
#### 功能概述
该文件实现了改进的斑翠鸟优化算法（IPKO），包括 Kent 混沌映射、种群初始化、目标函数定义和优化过程。

#### 代码块功能
- **球体函数 (`sphere_function`)**：定义目标函数。
- **Kent 映射 (`kent_map`)**：生成 Kent 混沌序列。
- **种群初始化 (`initialize_population_kent`)**：使用 Kent 映射初始化种群。
- **IPKO 算法实现 (`IPKO`)**：实现改进的 IPKO 优化算法，包括探索与开发阶段切换、捕食效率调整等。

### 4. `3D mapping.py`
#### 功能概述
该文件实现了三维地图构建、无人机集群约束与目标函数建立。

#### 代码块功能
- **三维地图构建 (`mountain_terrain`)**：生成山地地形的三维地图。
- **无人机集群约束函数**：
  - `route_length_cost`：计算航程约束成本。
  - `flight_height_cost`：计算飞行高度约束成本。
  - `inertia_distance_cost`：计算惯性距离约束成本。
  - `pitch_angle_cost`：计算俯仰角约束成本。
  - `multi_drone_collision_cost`：计算多无人机协同约束成本。
- **目标函数 (`objective_function`)**：综合考虑各种约束条件，计算目标函数值。

### 5. `ipkoz.py`
#### 功能概述
该文件实现了基于 IPKO 算法的无人机路径规划，包括环境定义、路径规划、优化过程和结果可视化。

#### 代码块功能
- **环境定义 (`Environment` 类)**：
  - `__init__`：初始化环境，包括地形障碍物。
  - `is_collision`：检查单个点或路径段是否与任何障碍物或威胁发生碰撞。
  - `check_inter_uav_collision`：检查无人机路径之间的碰撞。

- **路径规划 (`PathPlanner` 类)**：
  - `__init__`：初始化路径规划器，包括环境和无人机参数。
  - `generate_initial_path`：初始生成直线路径。
  - `calculate_path_length`：计算单个路径的总长度。
  - `calculate_fitness`：计算所有无人机路径集的适应度（成本）。

- **IPKO 算法实现 (`IPKO_Optimizer` 类)**：
  - `__init__`：初始化优化器，包括路径规划器、种群大小、最大迭代次数等参数。
  - `initialize_population`：初始化种群。
  - `run`：运行 IPKO 优化循环。
  - `calculate_population_center`：计算种群中心。
  - `calculate_target_center`：计算目标中心。

- **可视化函数 (`plot_results`)**：绘制 3D 环境、障碍物和规划路径。

## 四、运行步骤
1. 确保你已经安装了所需的依赖库。
2. 运行 `ipkoz-PCL.py` 文件，即可开始无人机路径规划的优化过程。

```bash
python ipkoz-PCL.py
```

## 五、注意事项
- 部分代码块中的函数体被省略，需要根据实际需求进行实现。
- 一些参数（如 PSO 相关参数）需要根据具体问题进行调整。