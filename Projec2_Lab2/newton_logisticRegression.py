#######################
# Author : Xinru Shan
# Date : 2018.10.17- 2018.10.26
# Student ID : 1160100626
# Email : sxr19980217@163.com
#######################

from numpy import *
import matplotlib.pyplot as plt
import numpy as np
import random
from numpy.linalg import cholesky

class newtonLR:
    """ Summary of class here.
        Train a logistic Regression model with Newton Method.
        less than 3 attributes: Plot and print the result.
        more than 3 attributes: Only print the result.
        Evaluate model with accuracy, precision and recall.
    """
    def __init__(self, epsilon, lamda):
        self.train_x = []       # 训练集x
        self.train_y = []       # 训练集y
        self.theta = []         # 待学习参数
        self.test_x = []        # 测试集x
        self.test_y = []        # 测试集y
        self.epsilon = epsilon  # 迭代阈值
        self.lamda = lamda      # 正则项系数
        self.loop_max = 3       # 最大循环次数
        self.dimension = 0      # 特征维数

    def loadData(self, path):
        """ load data from txt
            :param path: the path of file
            :return: none
        """
        with open(path)as f:
            lines = f.readlines()

        temp_x = []
        temp_y = []
        self.dimension = len(lines[0].split(',')) - 1
        for line in lines:
            read = line.strip().split(',')
            temp = [1]          # 在x矩阵的第0列加入1
            for i in range(0, self.dimension):
                temp.append(float(read[i]))
            temp_x.append(temp)
            temp_y.append(int(read[self.dimension]))
        # 将数据集按照7:3划分为训练集与测试集
        np.random.seed()
        self.test_x = random.sample(temp_x, round(len(temp_y)*0.3))
        for x in self.test_x:
            i = temp_x.index(x)
            self.test_y.append(temp_y[i])
        for item in temp_x:
            if item not in self.test_x:
                self.train_x.append(item)
                index = temp_x.index(item)
                self.train_y.append(temp_y[index])
        # 初始化theta为0
        self.theta = np.mat(np.zeros(self.dimension + 1))
        self.train_x = np.mat(self.train_x)
        self.train_y = np.mat(self.train_y).T
        self.test_x = np.mat(self.test_x)
        self.test_y = np.mat(self.test_y).T

    def accuracy(self):
        """ calculate accuracy of the model
            :return: accuracy
        """
        predict = (self.theta*self.test_x.T).T
        sum = 0
        for i in range(0, len(predict) - 1):
            if (predict[i] >= 0) and (self.test_y[i]) == 1:
                sum += 1
            elif predict[i] < 0 and self.test_y[i] == 0:
                sum += 1
        return sum/len(predict)

    def precision(self):
        """ calculate precision of the model
            :return: precision
        """
        predict = (self.theta * self.test_x.T).T
        TP = 0
        FP = 0
        for i in range(0, len(predict) - 1):
            if predict[i] >= 0 and self.test_y[i] == 1:
                TP += 1
            elif predict[i] >= 0 and self.test_y[i] == 0:
                FP += 1
        return float(TP/(FP + TP))

    def recall(self):
        """calculate recall of the model
            :return: recall
        """
        predict = (self.theta * self.test_x.T).T
        TP = 0
        FN = 0
        for i in range(0, len(predict) - 1):
            if predict[i] >= 0 and self.test_y[i] == 1:
                TP += 1
            elif predict[i] < 0 and self.test_y[i] == 1:
                FN += 1
        return float(TP / (FN + TP))

    def sigmod(self, input):
        """ calculate by f(z) = 1/(1+exp(z))
            :param input: z
            :return: f(z)
        """
        return 1.0/(1 + np.exp(-input))

    def normal_gradient(self, X, Y, theta):
        """ calculate gradient
        :param X: matrix x
        :param Y: matrix y
        :param theta: matrix theta
        :return: gradient
        """
        return (self.sigmod(theta*X.T).T - Y).T * X/len(Y)

    def regular_gradient(self, X, Y, theta):
        """ calculate gradient
            :param X: matrix x
            :param Y: matrix y
            :param theta: matrix theta
            :return: gradient with regularization
        """
        return (self.sigmod(theta*X.T).T - Y).T * X/len(Y) + (self.lamda * theta)/len(Y)

    def hessianMatrix(self, X, Y, theta):
        """calculate Hessian Matrix
        :param X: matrix x
        :param Y: matrix y
        :param theta: matrix theta
        :return: Hessian Matrix
        """
        h = self.sigmod(theta*X.T).T
        m = np.multiply(h, (1-h))
        M = X/len(Y)
        for i in range(0, self.dimension + 1):
            M[:, i] = np.multiply(M[:, i], m)
        return X.T * M

    def normal_newtonMethod(self):
        """ train a logistic regression model without regularization
            :return: learned theta
        """
        k = 0
        while k < self.loop_max:
            g = self.normal_gradient(self.train_x, self.train_y, self.theta)    # 计算梯度
            H = self.hessianMatrix(self.train_x, self.train_y, self.theta)      # 计算hessian矩阵
            if np.linalg.norm(g) < self.epsilon:
                break
            else:
                self.theta = self.theta - (H.I * g.T).T                         # 更新theta
                k = k + 1
            print(self.theta)
        return self.theta

    def regular_newtonMethod(self):
        """ train a logistic regression model with regularization
            :return: learned theta
        """
        k = 0
        while k < self.loop_max:
            g = self.regular_gradient(self.train_x, self.train_y, self.theta)
            regular = np.eye(len(self.theta[0]), k=0)
            regular[0, 0] = 0
            regular_m = (self.lamda * regular)/len(self.train_y)
            H = self.hessianMatrix(self.train_x, self.train_y, self.theta) + regular_m

            if np.linalg.norm(g) < self.epsilon:
                break
            else:
                self.theta = self.theta - (H.I * g.T).T
                k = k + 1
            print(self.theta)
        return self.theta

    def plotResult(self, type):
        """ plot result with 2-d figure
            :return: none
        """
        plt.title('classification result(' + type +')')
        for i in range(0, len(self.train_y)):
            if self.train_y[i] == 0:
                plt.plot(self.train_x[i, 1], self.train_x[i, 2], color='r', linestyle='', marker='.',)
            elif self.train_y[i] == 1:
                plt.plot(self.train_x[i, 1], self.train_x[i, 2], color='b', linestyle='', marker='.',)
        sort = sorted(list(array(self.train_x)[:, 1]))
        min = sort[0]
        max = sort[len(sort) - 1]
        x_decision = np.linspace(min, max, 10)
        y_decision = -self.theta[0, 0] / self.theta[0, 2] - (self.theta[0, 1] / self.theta[0, 2]) * x_decision
        plt.plot(x_decision, y_decision, color='g', linestyle='-', marker='', label=type+' decision')
        plt.legend(loc='upper right')
        plt.show()

def normal_run():
    """ steps of train a model
        :return: a logistic regression model
    """
    model = newtonLR(1e-2, 0)
    model.loadData('data.txt')
    model.normal_newtonMethod()
    print('-----------normal result-------------')
    print("accuracy = ", model.accuracy())
    print("precision = ", model.precision())
    print("recall = ", model.recall())
    print('------------------------------------')
    model.plotResult('normal')
    return model

def regular_run():
    """ steps of train a model with regularization
        :return: a logistic regression model
    """
    model = newtonLR(1e-2, 0)
    model.loadData('data.txt')
    model.regular_newtonMethod()
    print('----------- regularization result-------------')
    print("accuracy = ", model.accuracy())
    print("precision = ", model.precision())
    print("recall = ", model.recall())
    print('----------------------------------------------')
    model.plotResult('regularization')
    return model

def testIndependence():
    """ generate the 2-D Gaussian sample
    :return: none
    """
    mu1 = np.array([[1, 1]])
    sigma1 = np.array([[3, 0.5], [0, 3]])
    R1 = cholesky(sigma1)
    s1 = np.dot(np.random.randn(100, 2), R1) + mu1
    x1 = s1[:, 0]
    x2 = s1[:, 1]
    mu2 = np.array([[3, 3]])
    sigma2 = np.array([[3, 0.5], [0.8, 3]])
    R2 = cholesky(sigma2)
    s2 = np.dot(np.random.randn(100, 2), R2) + mu2
    x3 = s2[:, 0]
    x4 = s2[:, 1]
    with open('testIndependence.txt', 'w') as f:
        for i in range(0, len(x1)):
            f.write(str(x1[i]) + ',' + str(x2[i]) + ',' + '1\n')
            f.write(str(x3[i]) + ',' + str(x4[i]) + ',' + '0\n')
        f.close()

if __name__ == '__main__':
    # testIndependence()
    normal_model = normal_run()
    regular_model = regular_run()


