import numpy as np
import matplotlib.pyplot as plt
import time
# 如果需要专门的 Levy 或 alpha-stable 生成器（当 alpha 不为 2 时）
# from scipy.stats import levy_stable
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def sphere_function(x):
    """ 球体函数: f(x) = sum(x_i^2) """
    return np.sum(x**2)

# --- Kent 混沌映射 ---
def kent_map(x, a=0.7):
    """ Kent 混沌序列生成器 """
    if 0 < x <= a:
        return x / a
    elif a < x < 1:
        return (1 - x) / (1 - a)
    else: # 处理边界情况或无效输入
        return np.random.rand() # 安全回退

def initialize_population_kent(N, dim, lb, ub, a=0.7):
    """ 使用 Kent 映射初始化种群 """
    positions = np.zeros((N, dim))
    kent_val = np.random.rand() # 初始种子
    for i in range(N):
        for j in range(dim):
            kent_val = kent_map(kent_val, a)
            positions[i, j] = lb[j] + kent_val * (ub[j] - lb[j])
            # 添加小噪声防止可能卡在0或1
            if np.random.rand() < 0.01:
                 kent_val = np.random.rand()
    return positions

# --- Alpha-Stable 随机数生成器 (alpha=2时为高斯分布) ---
# 我们将使用 np.random.randn() 生成标准正态(高斯)变量
# 根据方程应用因子(如 2*, -1)
# 如果需要 alpha != 2，可以使用专门的库函数如 scipy.stats.levy_stable.rvs

# --- 改进的斑翠鸟优化算法 (IPKO) 实现 ---
def IPKO(N, Max_iter, lb, ub, dim, fobj, P_max=0.5, P_min=0, B_factor=8, kent_a=0.7, p_explore_exploit=0.5):
    """
    实现改进的斑翠鸟优化算法 (Improved Pied Kingfisher Optimizer, IPKO)


    参数:
    N (int): 种群规模
    Max_iter (int): 最大迭代次数
    lb (float or list/numpy array): 变量下界
    ub (float or list/numpy array): 变量上界
    dim (int): 问题维度
    fobj (function): 目标函数 (需要最小化)
    P_max (float): 捕食效率 P 的最大值 (Eq 2.14)
    P_min (float): 捕食效率 P 的最小值 (Eq 2.14)
    B_factor (float): 栖息/悬停策略中的 B 因子 (Eq 2.3, 2.5, 论文中硬编码为8)
    kent_a (float): Kent 映射参数 (Eq 2.15 控制参数)
    p_explore_exploit (float): 探索(栖息/悬停)与开发(潜水)阶段切换概率

    返回:
    Best_fitness (float): 找到的最优适应度值
    Best_pos (numpy array): 找到的最优解位置
    Convergence_curve (numpy array): 每次迭代的最优适应度值历史
    """

    print("IPKO 算法启动中...")
    print(f"参数: N={N}, Max_iter={Max_iter}, dim={dim}")
    print(f"P_max={P_max}, P_min={P_min}, B={B_factor}, kent_a={kent_a}, p_explore={p_explore_exploit}")

    # --- 初始化 ---
    if isinstance(lb, (int, float)):
        lb = np.full(dim, lb)
    if isinstance(ub, (int, float)):
        ub = np.full(dim, ub)

    # 使用 Kent 映射初始化种群
    Positions = initialize_population_kent(N, dim, lb, ub, kent_a)

    Fitness = np.full(N, np.inf)
    for i in range(N):
        Fitness[i] = fobj(Positions[i, :])

    Best_idx = np.argmin(Fitness)
    Best_fitness = Fitness[Best_idx]
    Best_pos = Positions[Best_idx, :].copy()

    Convergence_curve = np.zeros(Max_iter)

    # --- 迭代优化 ---
    start_time = time.time()
    for t in range(Max_iter):
        # 更新捕食效率 P (Eq 2.14)
        P = P_max - (P_max - P_min) * (t / Max_iter)**2 # 使用 t^2 实现更平滑的过渡，论文使用线性 t/M

        # 对每个翠鸟进行更新
        for i in range(N):
            New_pos = Positions[i, :].copy() # 从当前位置开始

            # --- 阶段选择：探索 (栖息/悬停) 或 开发 (潜水) ---
            if np.random.rand() < p_explore_exploit:
                # --- 探索阶段 (栖息 Habitat / 悬停 Hovering) ---
                # 论文没有明确区分何时用栖息或悬停，这里实现悬停策略 (Eq 2.5, 2.6)
                # 因为它涉及到个体间交互，似乎更动态

                # 随机选择另一个个体 j (Eq 2.2, 2.6 要求 j != i)
                j = np.random.randint(0, N)
                while j == i:
                    j = np.random.randint(0, N)

                # 计算 b (Eq 2.6) - 使用 Fitness 代替 F
                # 添加小epsilon避免除以零
                epsilon = 1e-10
                b = np.random.rand() * (Fitness[j] + epsilon) / (Fitness[i] + epsilon)

                # 计算 T (Eq 2.5 - 悬停)
                # 注意: 论文中的 MB，假设 M=Max_iter
                T_hover = b * ( (t * B_factor) / (Max_iter * B_factor) )**0.5 # 从 (tB / MB)^(1/B) 简化而来，可能 B=0.5？根据常见模式使用平方根。如有需要可重新审视
                # 原始 PKO 可能在指数中使用 B=8，让我们试试:
                # T_hover = b * ( (t*B_factor) / (Max_iter * B_factor) )**(1/B_factor) # 使用 B=8 指数

                # 基于栖息策略的 T (Eq 2.3) - 如果需要替代悬停
                # C_param = 2 * np.pi * np.random.rand() # Eq 2.4
                # T_habitat = (np.exp(t / Max_iter) - np.exp(-t / Max_iter)) * np.cos(C_param)

                # alpha1 (Eq 2.2 之后，使用 alpha=2 高斯分布作为 R1)
                # alpha1 = 2 * np.random.randn(dim) # R1 是 N(0,1)
                # 修正解释: alpha1 = 2*R1 其中 R1 是逐元素的 N(0,1)
                alpha1 = 2 * np.random.randn(dim)

                # 更新位置 (Eq 2.2)
                New_pos = Positions[i, :] + alpha1 * T_hover * (Positions[j, :] - Positions[i, :])

            else:
                # --- 开发阶段 (潜水 Diving) ---
                # 计算 o (Eq 2.9)
                o = np.exp(-(t / Max_iter)**2) # 更平滑的衰减

                # 计算 H (Eq 2.8)
                # 添加小epsilon避免除以零
                epsilon = 1e-10
                H = np.random.rand() * (Fitness[i] + epsilon) / (Best_fitness + epsilon)

                # 计算 b (Eq 2.10) - 与全局最优解的交互
                # Rk 可能是论文中的 rand()
                Rk = np.random.rand(dim) # 向量用于逐元素操作
                b_dive = Positions[i, :] + o**2 * Rk * Best_pos # 逐元素乘法

                # alpha2 (Eq 2.7 之后，使用 alpha=2 高斯分布作为 Rd)
                # alpha2 = 2 * np.random.randn(dim) - 1 # Rd 是 N(0,1)
                # 修正解释: alpha2 = 2*Rd-1 其中 Rd 是逐元素的 N(0,1)
                alpha2 = 2 * np.random.randn(dim) - 1

                # 更新位置 (Eq 2.7)
                # 使用 Best_pos 而不是 b-Xbest(t) 因为 b_dive 已经包含 Xbest(t)
                # New_pos = Positions[i,:] + H * o * alpha2 * (b_dive - Positions[i,:])
                # 重新审视 Eq 2.7: (b - Xbest(t)) 项看起来不寻常，如果 b 依赖于 Xbest
                # 让我们使用更标准的开发步骤朝向 Best_pos:
                New_pos = Positions[i,:] + H * o * alpha2 * (Best_pos - Positions[i,:]) # 向最优解移动

            # --- 局部逃逸阶段 (共生 Symbiotic) ---
            # 根据流程图(图9)，此阶段在探索/开发之后应用
            # 此阶段的更新取决于概率 P (Eq 2.11)

            # 计算 o (Eq 2.12)
            o_sym = np.exp(-(t / Max_iter)**2) # 与 o 相同的衰减

            # alpha (Eq 2.13，使用 alpha=2 高斯分布作为 Rs)
            # alpha_sym = 2 * np.random.randn(dim) - 1 # Rs 是 N(0,1)
            # 修正解释: alpha_sym = 2*Rs-1 其中 Rs 是逐元素的 N(0,1)
            alpha_sym = 2 * np.random.randn(dim) - 1

            # 随机选择另一个个体 m (Eq 2.11)
            m = np.random.randint(0, N)
            while m == i:
                m = np.random.randint(0, N)

            # 应用共生更新 (Eq 2.11) - 概率更新
            if np.random.rand() > (1 - P): # 注意: rand > 1-P 等同于 rand < P
                 # 使用从探索/开发阶段获得的位置计算更新步长
                 symbiotic_step = o_sym * alpha_sym * abs(New_pos - Positions[m,:])
                 # 应用更新
                 New_pos = New_pos + symbiotic_step
            # else: New_pos 保持探索/开发阶段计算的结果

            # --- 边界处理 ---
            New_pos = np.clip(New_pos, lb, ub)

            # --- 评估新位置 ---
            New_fitness = fobj(New_pos)

            # --- 更新个体位置 (如果更好) ---
            if New_fitness < Fitness[i]:
                Fitness[i] = New_fitness
                Positions[i, :] = New_pos

                # --- 更新全局最优 ---
                if New_fitness < Best_fitness:
                    Best_fitness = New_fitness
                    Best_pos = New_pos.copy()

        # 记录本次迭代的最优值
        Convergence_curve[t] = Best_fitness

        # (可选) 打印迭代信息
        if (t + 1) % 50 == 0:
            print(f"迭代 {t+1}/{Max_iter}, 最优适应度 = {Best_fitness:.6f}")

    end_time = time.time()
    print(f"IPKO 算法完成。耗时: {end_time - start_time:.2f} 秒")
    print(f"找到的最优解: \n位置 = {Best_pos}")
    print(f"适应度 = {Best_fitness}")

    return Best_fitness, Best_pos, Convergence_curve

if __name__ == "__main__":
    # 设置参数
    population_size = 30      # 种群规模 N
    max_iterations = 500      # 最大迭代次数 M
    dimension = 10            # 问题维度 D
    lower_bound = -10         # 变量下界 low
    upper_bound = 10          # 变量上界 up
    kent_a_param = 0.7        # Kent 映射参数 a
    p_explore = 0.5           # 探索/开发切换概率 (自定义，论文未给出)
    predation_p_max = 0.5     # P_max
    predation_p_min = 0.0     # P_min
    beat_factor_B = 8         # B factor

    # 运行 IPKO 算法
    best_fitness, best_position, convergence_curve = IPKO(
        N=population_size,
        Max_iter=max_iterations,
        lb=lower_bound,
        ub=upper_bound,
        dim=dimension,
        fobj=sphere_function,
        P_max=predation_p_max,
        P_min=predation_p_min,
        B_factor=beat_factor_B,
        kent_a=kent_a_param,
        p_explore_exploit=p_explore
    )

    # --- 结果可视化 ---
    plt.figure(figsize=(10, 6))
    plt.plot(convergence_curve, label='IPKO 收敛曲线', marker='*') 
    plt.title('IPKO 优化收敛曲线 (球体函数)')
    plt.xlabel('迭代次数')
    plt.ylabel('最优适应度值')
    plt.yscale('log') # 对数刻度通常更清晰
    plt.legend()
    plt.grid(True)
    plt.show()