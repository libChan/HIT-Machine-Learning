# -*- coding:utf-8 -*-
import random

import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
import math

#参数
num = 21                   # 样本组数
N = 5                       # 特征个数
loop_max = 100000           # 最大循环次数
epsilon = 1e-7              # 收敛条件的极小值
alpha = 0.7                 # 学习步长
loss_function = 0
loss=[]
lamda = 0.00002             # 加入正则项后的超参数

# 特征归一化
def scaling(feature, n):
    sum = feature.sum(axis=0)
    mean = sum/n
    s = feature[n-1] - feature[0]
    feature = (feature-mean)/s
    return feature

# 初始化矩阵为多项式回归
def initMatrix(x, N):
    m = []
    for i in range(0, N):
        a = x_discrete ** (i)
        m.append(a)
    x_scaling = np.mat(np.array(m)).T
    x = np.mat(np.array(m)).T
    for i in range(1, N):
        x_scaling[:, i] = scaling(x_scaling[:, i], num)
    x_scaling = np.mat(x_scaling)
    return x, x_scaling

# 梯度下降代码的核心部分
def gradient_descent(x,theta,y,x_scaling,alpha):
    error = 0
    count = 0
    while count < loop_max:
        count += 1
        h = x * theta
        for j in range(0, N):
            diff = 0
            diff = dot((h - y).T, x_scaling[:, j])      # 求偏导数
            theta[j] = theta[j] - alpha * (diff / num)  # 更新theta
        # 计算损失函数
        loss_function = 0
        h = x * theta
        for i in range(1, num):
            loss_function = loss_function + (y[i] - h[i]) ** 2
        loss_function = loss_function / (2 * num) + lamda*sum(pow(np.array(theta), 2))
        print(loss_function)
        # 判断收敛条件
        if abs(loss_function - error) < epsilon:
            break
        else:
            error = loss_function
    print(count)
    return theta

if __name__=="__main__":
    # 生成样本值
    x_discrete = np.linspace(0, 1, num)  # x样本值
    y = np.mat(np.sin(2 * math.pi * x_discrete)).T  # y样本值
    x_origin = np.linspace(0, 1, 100)
    y_origin = np.sin(2 * math.pi * x_origin)
    # 初始化theta向量
    np.random.seed(0)
    theta = np.mat(np.random.randn(N).reshape(-1, 1))
    # 在y方向上加入噪声
    mu = 0
    sigma = 0.1
    for i in range(len(y)):
        y[i] += random.gauss(mu, sigma)
    [x, x_scaling] = initMatrix(x_discrete, N)

    theta = gradient_descent(x, theta, y, x_scaling, alpha)

    predict = dot(x, theta)
    plt.plot(x_origin, y_origin, color='g', linestyle='-', marker='', label='original')
    plt.plot(x_discrete, y, color='r', linestyle='', marker='x', label='data')
    plt.plot(x_discrete, predict, color='b', linestyle='-', marker='.', label='fit')
    plt.legend(loc='upper right')
    plt.show()







