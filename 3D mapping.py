import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1.1 三维地图构建
def mountain_terrain(x, y, z0=0, M=3, h=np.ones(3), xm=np.ones(3), ym=np.ones(3), smx=np.ones(3), smy=np.ones(3)):
    terrain = z0
    for m in range(M):
        terrain += h[m] * np.exp(-((x - xm[m]) / smx[m])**2 - ((y - ym[m]) / smy[m])**2)
    return terrain

# 生成网格数据
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = mountain_terrain(X, Y)

# 绘制三维地图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='terrain')
plt.show()

# 1.2 无人机集群约束与目标函数建立

# 1.2.1 航程约束
def route_length_cost(path):
    d = len(path)
    l = [np.linalg.norm(path[i + 1] - path[i]) for i in range(d - 1)]
    T1 = np.sum(l)
    return T1

# 1.2.2 飞行高度约束
def flight_height_cost(path, terrain_function):
    D1 = 0
    C0 = 100
    for point in path:
        x, y, h = point
        z = terrain_function(x, y)
        if h <= z:
            D1 += (z - h) * C0
    return D1

# 1.2.3 惯性距离约束
def inertia_distance_cost(path, Lmin):
    D2 = 0
    C0 = 100
    for i in range(1, len(path)):
        L = np.linalg.norm(path[i] - path[i - 1])
        if L < Lmin:
            D2 += C0
    return D2

# 1.2.4 俯仰角约束
def pitch_angle_cost(path, max_theta):
    D3 = 0
    C0 = 100
    for i in range(1, len(path)):
        zi, xi, yi = path[i]
        zi_1, xi_1, yi_1 = path[i - 1]
        if zi != zi_1:
            theta = np.arctan(np.abs(zi - zi_1) / np.sqrt((xi - xi_1)**2 + (yi - yi_1)**2))
            if theta >= max_theta:
                D3 += C0
    return D3

# 1.2.5 多无人机协同约束
def multi_drone_collision_cost(paths, dmin):
    T3 = 0
    C0 = 100
    num_drones = len(paths)
    for i in range(num_drones):
        for j in range(i + 1, num_drones):
            for k in range(len(paths[i])):
                dij = np.linalg.norm(paths[i][k] - paths[j][k])
                if dij < dmin:
                    T3 += C0
    return T3

# 1.2.6 目标函数建立
def objective_function(paths, terrain_function, Lmin, max_theta, dmin, omega1=0.4, omega2=0.3, omega3=0.3):
    T1 = sum([route_length_cost(path) for path in paths])
    T2 = sum([flight_height_cost(path, terrain_function) + inertia_distance_cost(path, Lmin) + pitch_angle_cost(path, max_theta) for path in paths])
    T3 = multi_drone_collision_cost(paths, dmin)
    f = omega1 * T1 + omega2 * T2 + omega3 * T3
    return f

# 假设有两架无人机的路径
path1 = np.array([[0, 0, 10], [1, 1, 11], [2, 2, 12]])
path2 = np.array([[0, 1, 10], [1, 2, 11], [2, 3, 12]])
paths = [path1, path2]

# 参数设置
Lmin = 1
max_theta = np.pi / 4
dmin = 2

# 计算目标函数值
f = objective_function(paths, mountain_terrain, Lmin, max_theta, dmin)
print(f"目标函数值: {f}")
