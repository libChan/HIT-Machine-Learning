import tensorflow.examples.tutorials.mnist.input_data as input_data
from PIL import Image
from skimage import io
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2


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
    # k = percentage_d(eigVals, percentage)
    eig_val_sort = np.argsort(eig_vals)         # 对特征值进行排序
    eig_val_sort = eig_val_sort[:-(k+1):-1]     # 从升序排好的特征值，从后往前取k个
    feature = eig_vects[:, eig_val_sort]        # 返回主成分
    low_dimension = mean_scaling * feature      # 将原始数据投影到主成分上得到新的低维数据
    recon_data = (low_dimension * feature.T) + mean_vals    # 重构数据
    return low_dimension, recon_data


def mnist_test():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False) # 读入mnist数据集
    imgs = mnist.train.images
    labels = mnist.train.labels

    origin_imgs = []    # 取前1000张图片里的100个2
    for i in range(1000):
        if labels[i] == 2 and len(origin_imgs) < 100:
            origin_imgs.append(imgs[i])

    ten_origin_imgs = comb_imgs(origin_imgs, 10, 10, 28, 28, 'L')
    io.imsave('image/origin.png', ten_origin_imgs)
    low_d, recon_d = my_pca(np.array(origin_imgs), 1)

    recon_img = comb_imgs(recon_d, 10, 10, 28, 28, 'L')
    io.imsave('image/recon.png', recon_img)
    original = cv2.imread("image/origin.png")
    recon = cv2.imread("image/recon.png")
    compare_images(original, recon, 'mnist compare')


# 将array转到image
def array_to_img(array):
    array = array * 255
    new_img = Image.fromarray(array.astype(np.uint8))
    return new_img


# 拼图
def comb_imgs(origin_imgs, col, row, each_width, each_height, new_type):
    new_img = Image.new(new_type, (col * each_width, row * each_height))
    for i in range(len(origin_imgs)):
        each_img = array_to_img(np.array(origin_imgs[i]).reshape(each_width, each_width))
        new_img.paste(each_img, ((i % col) * each_width, (i // col) * each_width))
    return new_img


def psnr(img1, img2):
    mse = np.mean((img1-img2)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def compare_images(imageA, imageB, title):
    p = psnr(imageA, imageB)
    fig = plt.figure(title)
    plt.suptitle("PSNR: %.2f" % (p))

    # imageA
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    # imageB
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    mnist_test()