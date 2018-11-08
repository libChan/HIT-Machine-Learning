#######################
# Author : Xinru Shan
# Date : 2018.11.7
# Student ID : 1160100626
# Email : sxr19980217@163.com
#######################

import numpy as np
from matplotlib import pyplot as plt

class Kmeans():
    def __init__(self, k, epsilon):
        self.k = k
        self.epsilon = epsilon
        self.fetures = 0
        self.loop_max = 6
        self.centroids = []
        self.train_x = []
        self.labels = []

    def load_data(self, path):
        with open(path)as f:
            lines = f.readlines()
        for line in lines:
            temp = []
            self.fetures = len(line.split())
            for i in line.split():
                temp.append(float(i))
            self.train_x.append(temp)
        self.train_x = np.mat(self.train_x)
        self.centroids = np.mat(self.centroids)
        self.labels = [None] * len(self.train_x)
        f.close()

    def euclid_dist(self, v1, v2):
        return np.linalg.norm(v1-v2)

    def random_centroids(self):
        """
        init centroids randomly
        :return: initial centroids
        """
        sample_num, features = np.shape(self.train_x)
        self.centroids = np.zeros((self.k, features))
        for i in range(0, self.k):
            centroid = self.train_x[np.random.choice(range(sample_num))]
            self.centroids[i] = centroid
        return self.centroids

    def k_means(self):
        """
        clustering and update centroids
        :return: clustering result
        """
        centroids = self.random_centroids()
        count = 0
        for i in range(0, self.loop_max):
            count += 1
            self.get_labels()
            former_centroids = centroids
            centroids = self.update_centroids()

            # diff = centroids - former_centroids
            # if diff.any() < self.epsilon:
            #     break
        print(count)
        print('-----------------result----------------')
        print("centroids: ", self.centroids)
        print('---------------------------------------')
        return self.labels

    def update_centroids(self):
        """
        update centroids
        :return: the updated centroids
        """
        for i in range(0, self.k):
            sum = 0
            num = 0
            for j in range(0, len(self.train_x)):
                if self.labels[j] == i:
                    sum += self.train_x[j]
                    num += 1
            for m in range(0, self.fetures):
                sum[0, m] = float(sum[0, m]/num)
            self.centroids[i] = sum
        return self.centroids

    def get_labels(self):
        """
        for each sample, find the shortest distance
        :return: none
        """
        length = len(self.train_x)
        for i in range(0, length):
            min_dist = 10000
            min_index = 0
            for j in range(0, len(self.centroids)):
                if self.euclid_dist(self.train_x[i], self.centroids[j]) < min_dist:
                    min_dist = self.euclid_dist(self.train_x[i], self.centroids[j])
                    min_index = j
            self.labels[i] = min_index

    def plot_result(self):
        for i in range(0, len(self.train_x)):
            if self.labels[i] == self.k - 1:
                plt.plot(self.train_x[i, 0], self.train_x[i, 1], marker='.', color='y')
            elif self.labels[i] == self.k - 2:
                plt.plot(self.train_x[i, 0], self.train_x[i, 1], marker='.', color='deepskyblue')
            elif self.labels[i] == self.k - 3:
                plt.plot(self.train_x[i, 0], self.train_x[i, 1], marker='.', color='g')
            elif self.labels[i] == self.k - 4:
                plt.plot(self.train_x[i, 0], self.train_x[i, 1], marker='.', color='#000000')
            elif self.labels[i] == self.k - 5:
                plt.plot(self.train_x[i, 0], self.train_x[i, 1], marker='.', color='deeppink')
            elif self.labels[i] == self.k - 6:
                plt.plot(self.train_x[i, 0], self.train_x[i, 1], marker='.', color='palegreen')

        for j in range(0, self.k):
            plt.plot(self.centroids[j, 0], self.centroids[j, 1], marker='*', color='r')

        plt.show()

def create_data():
    np.random.seed()
    s1 = np.random.normal(3, 0.4, 50)

    s2 = np.random.normal(3.2, 0.3, 50)
    s3 = np.random.normal(4, 0.5, 50)
    s4 = np.random.normal(4.3, 0.8, 50)
    s5 = np.random.normal(5.4, 0.5, 50)
    s6 = np.random.normal(5.7, 0.4, 50)
    with open('testGaussian.txt', 'w') as f:
        for i in range(0, len(s1)):
            f.write(str(s1[i]) + ' ' + str(s2[i]) + '\n')
            f.write(str(s3[i]) + ' ' + str(s4[i]) + '\n')
            f.write(str(s5[i]) + ' ' + str(s6[i]) + '\n')
        f.close()


if __name__ == "__main__":
    create_data()
    model = Kmeans(3, 1e-4)
    model.load_data('testGaussian.txt')
    x = model.train_x
    model.k_means()
    model.plot_result()


