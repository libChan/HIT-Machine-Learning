# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
import math


#参数
num = 11            # 样本组数
N = 6               # 特征个数
epsilon = 1e-3      # 收敛条件的极小值

# 初始化矩阵
def initMatrix(x, y, N):
    m = []
    for i in range(0, N):
        a = x**(i)
        m.append(a)
    x_new = np.mat(np.array(m)).T
    A = x_new.T*x_new
    b = x_new.T*y
    return A, b, x_new

if __name__=="__main__":
    # 生成样本值
    x_discrete = np.linspace(0, 1, num)  # x样本值
    y = np.mat(np.sin(2 * math.pi * x_discrete)).T  # y样本值
    x_origin = np.linspace(0, 1, 100)
    y_origin = np.sin(2 * math.pi * x_origin)
    # 初始化theta向量
    theta = np.mat(np.zeros(N).reshape(-1, 1))
    k = 0
    [A, b, x_new] = initMatrix(x_discrete, y, N)
    r = b - A*theta
    p = r

    # 共轭梯度的核心部分
    while k <= N:
        r_before = r
        a = dot(r.T, r)/dot(p.T, dot(A, p))
        print(a)
        theta = theta + dot(p, a)
        r = r - A*p*a
        if np.linalg.norm(r) < epsilon:
            break
        else:
            beta = dot(r.T, r)/dot(r_before.T, r_before)
            p = r + p*beta
            k = k + 1

    print(theta)
    predict = x_new*theta
    plt.plot(x_origin, y_origin, color='g', linestyle='-', marker='', label='original')
    plt.plot(x_discrete, y, color='r', linestyle='', marker='x', label='data')
    plt.plot(x_discrete, predict, color='b', linestyle='-', marker='.', label='fit')
    plt.legend(loc='upper right')
    plt.show()