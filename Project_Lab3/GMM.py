#######################
# Author : Xinru Shan
# Date : 2018.11.7
# Student ID : 1160100626
# Email : sxr19980217@163.com
#######################

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GMM():
    def __init__(self, path, k, epsilon, loop_max):
        assert k > 1
        self.k = k
        self.x = []
        self.loop_max = loop_max
        self.epsilon = epsilon
        self.dimension = 0
        self.sample = 0
        self.alpha = []
        self.mu = []
        self.cov = []
        self.x_array = self.load_data(path)

    def load_data(self, path):
        x_array = np.loadtxt(path)
        self.x = np.matrix(x_array, copy=True)
        self.sample = self.x.shape[0]
        self.alpha = np.array([1.0 / self.k] * self.k)
        self.dimension = self.x.shape[1]
        self.mu = np.random.rand(self.k, self.dimension)
        self.cov = np.array([np.eye(self.dimension)] * self.k)

        return x_array

    def scaling(self):
        for i in range(0, self.dimension):
            min = self.x[:, i].min()
            max = self.x[:, i].max()
            self.x[:, i] = (self.x[:, i] - min)/(max - min)
        return self.x

    def phi(self, X, mu, cov):
        norm = multivariate_normal(mean=mu, cov=cov)
        return norm.pdf(X)

    def e_step(self, mu, cov, alpha):
        """
        e-step of EM algorithm
        :param mu: mean
        :param cov: Covariance matrix
        :param alpha: probability of  each model
        :return: responsibility for each sample
        """
        # gamma为响应度矩阵，行代表样本，列表示对第k个模型
        gamma = np.mat(np.zeros((self.sample, self.k)))
        likelyhood = np.mat(np.zeros((self.sample, 1)))
        probility = np.zeros((self.sample, self.k))
        # 计算k个模型中，样本的概率
        for k in range(0, self.k):
            probility[:, k] = self.phi(self.x, mu[k], cov[k])
        probility = np.mat(probility)
        # 计算响应度矩阵
        for k in range(0, self.k):
            gamma[:, k] = alpha[k] * probility[:, k]
        for i in range(0, self.sample):
            likelyhood[i, :] = np.sum(gamma[i, :])
            gamma[i, :] = gamma[i, :] / np.sum(gamma[i, :])

        return gamma, likelyhood

    def m_step(self, gamma):
        """
        m-step of EM algorithm
        :param gamma: responsibility for each sample
        :return: the update mu, cov, alpha
        """
        mu = np.zeros((self.k, self.dimension))
        alpha = np.zeros(self.k)
        cov = []

        for k in range(self.k):
            gamma_sum = np.sum(gamma[:, k])     # 计算响应度之和
            # 更新mu
            for d in range(self.dimension):
                mu[k, d] = np.sum(np.multiply(gamma[:, k], self.x[:, d])) / gamma_sum
            # 更新cov
            cov_k = np.mat(np.zeros((self.dimension, self.dimension)))
            for i in range(self.sample):
                cov_k += gamma[i, k] * (self.x[i] - mu[k]).T * (self.x[i] - mu[k]) / gamma_sum
            cov.append(cov_k)
            # 更新 alpha
            alpha[k] = gamma_sum / self.sample
        cov = np.array(cov)
        return mu, cov, alpha


    def gmm_em(self):
        """
        GMM algorithm, estimate mu, cov, alpha of GMM
        :return: estimate of mu, cov, alpha
        """
        self.x = self.scaling()
        oldlikelyhood = 1
        for i in range(0, self.loop_max):
            gamma, likelyhood = self.e_step(self.mu, self.cov, self.alpha)  # e-step
            print("likelyhood: ", np.log(np.sum(likelyhood)))
            self.mu, self.cov, self.alpha = self.m_step(gamma)  # m-step
            if np.abs(np.sum(likelyhood-oldlikelyhood)) < self.epsilon:
                print(i)
                break
            else:
                oldlikelyhood = likelyhood
        print('------------------result-------------------')
        print("mu:", self.mu)
        print("cov:", self.cov)
        print("alpha:", self.alpha)
        print('-------------------------------------------')
        return self.mu, self.cov, self.alpha

    def plot_result(self):
        N = self.x.shape[0]
        gamma = self.e_step(self.mu, self.cov, self.alpha)[0]
        category = gamma.argmax(axis=1).flatten().tolist()[0]

        class1 = np.array([self.x_array[i] for i in range(N) if category[i] == 0])
        class2 = np.array([self.x_array[i] for i in range(N) if category[i] == 1])
        class3 = np.array([self.x_array[i] for i in range(N) if category[i] == 2])
        plt.plot(class1[:, 0], class1[:, 1], marker='.', linestyle='', color='y')
        plt.plot(class2[:, 0], class2[:, 1], marker='.', linestyle='', color='deepskyblue')
        plt.plot(class3[:, 0], class3[:, 1], marker='.', linestyle='', color='g')
        plt.legend(loc="best")
        plt.show()

def create_data():
    np.random.seed()
    s1 = np.random.normal(3, 0.4, 50)
    s2 = np.random.normal(3.2, 0.3, 50)
    s3 = np.random.normal(4, 0.5, 50)
    s4 = np.random.normal(4.3, 0.8, 50)
    s5 = np.random.normal(6, 1, 50)
    s6 = np.random.normal(5.7, 0.8, 50)
    with open('testGaussian.txt', 'w') as f:
        for i in range(0, len(s1)):
            f.write(str(s1[i]) + ' ' + str(s2[i]) + '\n')
            f.write(str(s3[i]) + ' ' + str(s4[i]) + '\n')
            f.write(str(s5[i]) + ' ' + str(s6[i]) + '\n')
        f.close()


if __name__ == "__main__":
    # 高斯生成数据聚类结果，每次运行具有随机性
    create_data()
    model = GMM('testGaussian.txt', 3, 1e-6, 500)
    model.gmm_em()
    model.plot_result()

    # UCI数据测试，6个特征，3分类
    model2 = GMM('seeds_dataset_X.txt', 3, 1e-4, 500)
    model2.gmm_em()
    with open('result.txt', 'w') as f:
        f.write(str(model2.mu))
        f.write(str(model2.cov))
        f.write(str(model2.alpha))
    f.close()

