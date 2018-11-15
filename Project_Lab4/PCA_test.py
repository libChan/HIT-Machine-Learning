import numpy as np
from sklearn.decomposition import PCA


def percentage_d(eigVals, percentage):
    eig_vals_sort = np.sort(eigVals)
    eig_vals_sort = eig_vals_sort[-1::-1]
    eig_vals_sum = sum(eig_vals_sort)
    dimension = 0
    temp = 0
    for i in range(0, len(eig_vals_sort)):
        temp += eig_vals_sort[i]
        dimension += 1
        if temp >= eig_vals_sum * percentage:
            return dimension


# 行表示样本，列表示特征维度
def my_pca(X, k):
    mean_vals = np.mean(X, axis=0)  # 对每一维求均值
    mean_scaling = X - mean_vals    # 每一维零均值化
    cov = np.cov(mean_scaling, rowvar=0)    # 计算方差
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov))    # 计算特征值和特征向量

    eig_val_sort = np.argsort(eig_vals)         # 对特征值进行排序
    eig_val_sort = eig_val_sort[:-(k+1):-1]     # 从升序排好的特征值，从后往前取k个
    feature = eig_vects[:, eig_val_sort]        # 返回主成分
    low_dimension = mean_scaling * feature      # 将原始数据投影到主成分上得到新的低维数据
    recon_data = (low_dimension * feature.T) + mean_vals    # 重构数据
    return low_dimension, recon_data


def create_data():
    np.random.seed()
    s1 = np.random.normal(3, 2, 50).reshape(50, 1)
    s2 = np.random.normal(3.5, 2.4, 50).reshape(50, 1)
    s3 = np.random.normal(8, 0.05, 50).reshape(50, 1)
    s = np.hstack((s1, s2))
    x = np.hstack((s, s3))

    low_data, recon_data = my_pca(np.mat(x), 1)
    pca = PCA(n_components=1)
    print('--------------第一行为my_pca结果，第二行为sklearn结果-----------------')
    result = np.vstack((-low_data.T, pca.fit_transform(np.mat(x)).T))
    print(result)
    print('--------------------------------------------------------------------')


if __name__ == "__main__":
    create_data()