import numpy as np
import random

# 定义Kent映射
def kent_map(x, a=0.7):
    if 0 < x <= a:
        return x / a
    else:
        return (1 - x) / (1 - a)
# 生成初始种群
def initialize_population(pop_size, dim, a=0.7):
    population = []
    for _ in range(pop_size):
        x = random.random()
        individual = []
        for _ in range(dim):
            x = kent_map(x, a)
            individual.append(x)
        population.append(individual)
    return np.array(population)

# 定义适应度函数（这里简化为目标函数）
def fitness_function(path, omega1=0.4, omega2=0.3, omega3=0.3):
    # 假设T1, T2, T3为约束条件的评估值
    T1 = np.sum(np.abs(np.diff(path)))  # 简单示例：总路径长度
    T2 = 0  # 可根据实际情况计算
    T3 = 0  # 可根据实际情况计算
    return omega1 * T1 + omega2 * T2 + omega3 * T3

# α-stable分布（这里简化为高斯分布）
def alpha_stable_distribution():
    return np.random.normal()

# 斑翠鸟优化算法主函数
def ipko(pop_size, dim, max_iter):
    # 初始化种群
    population = initialize_population(pop_size, dim)
    fitness_values = np.array([fitness_function(ind) for ind in population])
    best_index = np.argmin(fitness_values)
    best_solution = population[best_index]
    best_fitness = fitness_values[best_index]

    for iter in range(max_iter):
        # 动态聚类策略（简化）
        sorted_indices = np.argsort(fitness_values)
        sub_populations = np.array_split(sorted_indices, 3)  # 划分3个子种群

        for sub_pop in sub_populations:
            for i in sub_pop:
                individual = population[i]
                # 栖息与悬停策略
                T = 1 - iter / max_iter
                if random.random() < T:
                    new_individual = individual + alpha_stable_distribution()
                else:
                    # 潜水策略
                    new_individual = individual + (best_solution - individual) * alpha_stable_distribution()

                # 共生阶段
                if random.random() < 0.1:
                    new_individual += np.random.uniform(-0.1, 0.1, dim)

                new_fitness = fitness_function(new_individual)
                if new_fitness < fitness_values[i]:
                    population[i] = new_individual
                    fitness_values[i] = new_fitness
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        best_solution = new_individual
        # 周期性重组子种群（简化）
        if iter % 10 == 0:
            population = population[np.random.permutation(pop_size)]

    return best_solution, best_fitness

pop_size = 50
dim = 10
max_iter = 100
best_path, best_cost = ipko(pop_size, dim, max_iter)
print(f"Best path: {best_path}")
print(f"Best cost: {best_cost}")
