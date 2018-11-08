#######################
# Author : Xinru Shan
# Date : 2018.10.17- 2018.10.26
# Student ID : 1160100626
# Email : sxr19980217@163.com
#######################
from numpy import *
import matplotlib.pyplot as plt
import math
import numpy as np
import random

class logisticModel:
    """Summary of class here.
        Train a logistic Regression model with Gradient Descent.
        less than 3 attributes: Plot and print the result.
        more than 3 attributes: Only print the result.
        Evaluate model with accuracy, precision and recall.
    """

    def __init__(self, alpha, epsilon, lamda):
        self.train_x = []       # 训练集x
        self.train_y = []       # 训练集y
        self.theta = []         # 待学习参数
        self.test_x = []        # 测试集x
        self.test_y = []        # 测试集y
        self.alpha = alpha      # 学习速率
        self.epsilon = epsilon  # 迭代阈值
        self.lamda = lamda      # 正则项系数
        self.loop_max = 1000    # 最大循环次数
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
            temp = [1]      # 在x矩阵的第0列加入1
            for i in range(0, self.dimension):
                temp.append(float(read[i]))
            temp_x.append(temp)
            temp_y.append(int(read[self.dimension]))

        # 将数据集按照7:3划分为训练集与测试集
        np.random.seed()
        self.test_x = random.sample(temp_x, round(len(temp_y) * 0.3))
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

    def generateData(self):
        mu1 = 0.5
        sigma1 = 0.1
        mu2 = 0.8
        sigma2 = 0.4
        np.random.seed(0)
        x1 = np.random.normal(mu1, sigma1, 50)
        np.random.seed(5)
        x2 = np.random.normal(mu1, sigma2, 50)
        np.random.seed(10)
        x3 = np.random.normal(mu2, sigma1, 50)
        np.random.seed(15)
        x4 = np.random.normal(mu2, sigma2, 50)

        for i in range(0, 2 * len(x1)):
            tmp = []
            if i < len(x1):
                tmp.append(1)
                tmp.append(x1[i])
                tmp.append(x2[i])
                self.train_x.append(tmp)
                self.train_y.append(1)
            elif i >= len(x1):
                tmp.append(1)
                tmp.append(x3[i-len(x1)])
                tmp.append(x4[i-len(x1)])
                self.train_x.append(tmp)
                self.train_y.append(0)
        # 初始化theta为0
        self.dimension = 2
        self.theta = np.mat(np.zeros(self.dimension + 1))
        self.train_x = np.mat(self.train_x)
        self.train_y = np.mat(self.train_y).T


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

    def normal_lossFunction(self, theta, x_in, y_in):
        """ calculate loss
        :param theta: matrix theta
        :param x_in: matrix x
        :param y_in: matrix y
        :return: value of loss
        """
        sum = 0
        for i in range(0, len(y_in)):
            # sum += y_in[i]*math.log(self.sigmod(self.theta*self.train_x[i].T)) + (1-y_in[i])*math.log(1-self.sigmod(self.theta*self.train_x[i].T))
            sum += y_in[i]*math.log(self.sigmod(theta*x_in[i].T)) + (1 - y_in[i])*math.log(1 - self.sigmod(theta*x_in[i].T))
        return -sum/len(y_in)

    def regualr_lossFunction(self, theta, x_in, y_in):
        """ calculate loss with regularization
                :param theta: matrix theta
                :param x_in: matrix x
                :param y_in: matrix y
                :return: value of loss with regularization
                """
        normal = self.normal_lossFunction(theta, x_in, y_in)
        sum = 0
        for i in range(0, len(theta[0])):
            sum += theta[0, i] * theta[0, i]
        return normal + (self.lamda * sum)/(2 * len(self.train_y))

    def normal_gradientDescent(self):
        """ train a logistic regression model
            :return: learned theta
        """
        count = 0
        error = 0
        while count < self.loop_max:
            count += 1
            # 更新theta
            for j in range(0, self.dimension + 1):
                diff = 0
                for i in range(0, len(self.train_y)):
                    diff += (self.sigmod(self.theta * self.train_x[i].T) - self.train_y[i]) * self.train_x[i, j]
                self.theta[0, j] = self.theta[0, j] - self.alpha * diff
            print('------')
            print(self.theta)
            print('-------')
            # 计算损失函数
            loss_function = self.normal_lossFunction(self.theta, self.train_x, self.train_y)
            if abs(loss_function - error) < self.epsilon:
                break
            else:
                error = loss_function
        print(self.theta)
        print("loop count: ", count)
        return self.theta

    def regular_gradientDescent(self):
        """ train a logistic regression model with  regularization
        :return: learned theta
        """
        count = 0
        error = 0
        while count < self.loop_max:
            count += 1
            # 更新theta
            for j in range(0, self.dimension + 1):
                diff = 0
                for i in range(0, len(self.train_y)):
                    diff += (self.sigmod(self.theta * self.train_x[i].T) - self.train_y[i]) * self.train_x[i, j]
                if j == 0:
                     self.theta[0, j] = self.theta[0, j] - self.alpha * diff
                else:
                    regular = (self.lamda*self.theta[0, j])/len(self.train_y)
                    self.theta[0, j] = self.theta[0, j] - self.alpha * (diff + regular)
            # 计算损失函数
            loss_function = self.regualr_lossFunction(self.theta, self.train_x, self.train_y)
            if abs(loss_function - error) < self.epsilon:
                break
            else:
                error = loss_function
        print(self.theta)
        print("loop count: ", count)
        return self.theta

    def plotResult(self):
        """plot result with 2-d figure
        :return: none
        """
        for i in range(0, len(self.train_y)):
            if self.train_y[i] == 0:
                plt.plot(self.train_x[i, 1], self.train_x[i, 2], color='r', linestyle='', marker='.',)
            elif self.train_y[i] == 1:
                plt.plot(self.train_x[i, 1], self.train_x[i, 2], color='b', linestyle='', marker='.',)
        sort = sorted(list(array(self.train_x)[:, 1]))
        min = sort[0]
        max = sort[len(sort) - 1]
        x_decision = np.linspace(min, max, 20)
        y_decision = -self.theta[0, 0]/self.theta[0, 2] - (self.theta[0, 1]/self.theta[0, 2])*x_decision
        plt.plot(x_decision, y_decision, color='g', linestyle='-', marker='', label='decision')
        plt.legend(loc='upper right')
        plt.show()

def normal_run():
    """ steps of train a model
    :return: a logistic regression model
    """
    model = logisticModel(0.001, 1e-4, 0.1)
    model.loadData('data.txt')
    model.normal_gradientDescent()
    print('-----------normal result-------------')
    print("accuracy = ", model.accuracy())
    print("precision = ", model.precision())
    print("recall = ", model.recall())
    print('------------------------------------')
    model.plotResult()
    return model

def regular_run():
    """ steps of train a model with regularization
        :return: a logistic regression model with  regularization
    """
    model = logisticModel(0.001, 1e-4, 0.1)
    model.loadData('data.txt')
    model.regular_gradientDescent()
    print('----------- regularization result-------------')
    print("accuracy = ", model.accuracy())
    print("precision = ", model.precision())
    print("recall = ", model.recall())
    print('----------------------------------------------')
    model.plotResult()
    return model

if __name__ == '__main__':
    normal_model = normal_run()
    regular_model = regular_run()




