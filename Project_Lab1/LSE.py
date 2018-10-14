# -*- coding:utf-8 -*-
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import dot
import math

degree = 10

def initMatrix(x, N):
    m = []
    for i in range(0, N):
        a = x ** (i)
        m.append(a)
    x_new = np.mat(np.array(m)).T
    return x_new

def initData(x, num):
    #训练集与测试集的比例为7:3
    x_train = np.mat(random.sample(x, round(num*0.7)))
    y_train = np.sin(2* math.pi *x_train)
    x_test = []
    for item in x:
        if item not in x_train:
            x_test.append(item)
    x_test = np.mat(x_test)
    y_test = np.sin(2*math.pi*x_test)
    return x_train, y_train, x_test, y_test

#最小二乘法
def LSE_normal(A, B, n):
    theta = dot(dot(inv(dot(A.T, A)), A.T), B)
    return theta

#加正则项的最小二乘法
def LSE_regular(A, B, n, lamda):
    I = np.mat(np.eye(n, k=0))
    I[0, :] = 0
    theta = dot(dot(inv(dot(A.T, A)+I*lamda), A.T), B)
    return theta

def LOSSNormal(predict, y, N):
    return pow(sum(pow(np.array(predict-y), 2))/(N*N), 1/2)

def LOSSRegular(predict, y, theta, N, lamda):
    return pow((sum(pow(np.array(predict-y), 2)) + lamda*sum(pow(np.array(theta), 2)))/(N*N), 1/2)


def regularResult(x_train, y_train, x_test, y_test, N, lamda):
    #由训练集求解
    A = initMatrix(np.array(x_train), N)
    B = y_train.T
    theta_regular = LSE_regular(A, B, N, lamda)
    predict_regular = dot(A, theta_regular)
    loss_regular = LOSSRegular(predict_regular, B, theta_regular, N, lamda)
    lossTrain_regular.append(loss_regular)

    #计算测试集的损失函数
    A_test = initMatrix(np.array(x_test), N)
    predict_regular = dot(A_test, theta_regular)
    loss_regular = LOSSRegular(predict_regular, y_test.T, theta_regular, N, lamda)
    lossTest_regular.append(loss_regular)

    #画出拟合曲线
    A_plot = initMatrix(np.array(x_origin), N)
    l = dot(A_plot, theta_regular)
    plt.plot(x_origin, y_origin, color='g', linestyle='-', marker='', label='original')
    plt.plot(x_train, y_train, color='r', linestyle='', marker='x', label='data')
    plt.plot(x_origin, l, color='b', linestyle='-', marker='', label='fit_normal')
    plt.show()


def normalResult(x_train, y_train, x_test,y_test, N):
    A = initMatrix(np.array(x_train), N)
    B = y_train.T
    theta_normal = LSE_normal(A, B, N)
    predict_normal = dot(A, theta_normal)
    loss_normal = LOSSNormal(predict_normal, B, N)
    lossTrain_normal.append(loss_normal)

    # 计算测试集的损失函数
    A_test = initMatrix(np.array(x_test), N)
    predict_normal = dot(A_test, theta_normal)
    loss_normal = LOSSNormal(predict_normal, y_test.T, N)
    lossTest_normal.append(loss_normal)

    # 画出拟合曲线
    A_plot = initMatrix(np.array(x_origin), N)
    l = dot(A_plot, theta_normal)
    plt.plot(x_origin, y_origin, color='g', linestyle='-', marker='', label='original')
    plt.plot(x_train, y_train, color='r', linestyle='', marker='x', label='data')
    plt.plot(x_origin, l, color='b', linestyle='-', marker='', label='fit_normal')
    plt.show()

if __name__=="__main__":
    lamda = 0.002
    # 得到离散化数据
    num = 11
    x = np.linspace(0, 1, num)
    x_origin = np.linspace(0, 1, 100)
    y_origin = np.sin(2 * math.pi * x_origin)
    # 初始化数据集，形成训练集和测试集
    [x_train, y_train, x_test, y_test] = initData(list(x), num)
    # 向数据中添加噪声
    mu = 0
    sigma = 0.1
    y_train = y_train.T
    y_test = y_test.T
    for i in range(len(y_train)):
        y_train[i] += random.gauss(mu, sigma)
    for i in range(len(y_test)):
        y_test[i] += random.gauss(mu, sigma)
    y_train = y_train.T
    y_test = y_test.T

    lossTrain_normal = []
    lossTrain_regular = []
    lossTest_normal = []
    lossTest_regular = []
    #对不同阶数，有无正则项进行测试
    for j in range(1, degree+2):
        # A*theta = B
        normalResult(x_train, y_train, x_test, y_test, j)
        # regularResult(x_train, y_train, x_test, y_test, j, lamda)

    M = np.linspace(0, degree, degree+1)
    # 非正则项画图
    plt.plot(M, lossTest_normal, color='b', linestyle='-', marker='x', label='test')
    plt.plot(M, lossTrain_normal, color='r', linestyle='-', marker='o', label='train')
    # 正则项画图
    # plt.plot(M, lossTest_regular, color='b', linestyle='-', marker='x', label='test')
    # plt.plot(M, lossTrain_regular, color='r', linestyle='-', marker='o', label='train')
    plt.legend(loc='upper right')
    plt.show()

