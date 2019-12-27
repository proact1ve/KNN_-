#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author:  suming
@Python Version:  python 3.7.4
@File:  Version: final
@Time:  2019/12/25
"""

import numpy as np
import operator
from os import listdir
import matplotlib.pyplot as plt

def file_to_matrix(filename):
    """
     函数作用：从文件中读入训练数据，并存储为矩阵
    :param filename:文件名字符串
    :return:训练样本矩阵和类标签向量
    """
    # 打开文件
    fr = open(filename)
    # 读取文件内容
    array_lines = fr.readlines()
    # 得到文件行数
    number_of_lines = len(array_lines)
    # 返回解析后的数据
    return_mat = np.zeros((number_of_lines, 3))
    # 定义类标签向量
    class_label_vector = []
    # 行索引值
    index = 0
    for line in array_lines:
        # 去掉 回车符号
        line = line.strip()
        # 用\t分割每行数据
        list_from_line = line.split('\t')
        # 选取前3个元素，将它们存储到特征矩阵中
        return_mat[index, :] = list_from_line[0:3]
        # 把该样本对应的标签放至标签向量，顺序与样本集对应。
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def auto_norm(data_set):
    """
    该函数可以自动将数字特征值转化为0到1的区间,即归一化训练数据
    """
    # 获取数据集中每列的最小数值
    min_vals = data_set.min(0)
    # 获取数据集中每列的最大数值
    max_vals = data_set.max(0)
    # 最大值与最小的差值
    ranges = max_vals - min_vals
    # 创建一个全0矩阵，用于存放归一化后的数据
    norm_data_set = np.zeros(np.shape(data_set))
    # 返回data_set的行数
    m = data_set.shape[0]
    # 原始数据值减去最小值
    norm_data_set = data_set - np.tile(min_vals, (m, 1))
    # 除以最大和最小值的差值,得到归一化数据
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
    # 返回归一化数据，最大值与最小的差值，每列的最小数值
    return norm_data_set, ranges, min_vals

# 将图像数据转换为（1，1024）向量
def img_to_vector(filename):
    """
        将图像转换为向量：该函数创建1×1024的NumPy数组，然后打开给定的文件，
        循环读出文件的前32行，并将每行的头32个字符值存储在NumPy数组中，最后返回数组。
    """
    # 创建1×1024的NumPy数组
    return_vect = np.zeros((1, 1024))
    # 打开给定的文件名
    file = open(filename)
    # 循环读出文件的前32行
    for i in range(32):
        lineStr = file.readline()
        # 将每行的头32个字符值存储在NumPy数组
        for j in range(32):
            return_vect[0, 32 * i + j] = int(lineStr[j])
    #返回数组
    return return_vect


# kNN分类器
def classifier(input_data, data_set, labels_set, k):
    """
        函数作用：使用k-近邻算法将每组数据划分到某个类中
        :param input_data:用于分类的输入数据(测试集)
        :param data_set:输入的训练样本集
        :param labels_set:训练样本标签
        :param k:用于选择最近邻居的数目，即kNN算法参数,选择距离最小的k个点
        :return:返回分类结果
        """
    # data_set.shape[0]返回训练样本集的行数
    dataSetSize = data_set.shape[0]
    # 在列方向上重复input_data，1次，行方向上重复input_data，data_set_size次
    diffMat = np.tile(input_data, (dataSetSize, 1)) - data_set
    # diff_mat：输入样本与每个训练样本的差值,然后对其每个x和y的差值进行平方运算
    sqDiffMat = diffMat ** 2
    # 按行进行累加，axis=1表示按行
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方运算，求出距离
    distances = sqDistances ** 0.5
    # 返回distances中元素从小到大排序后的索引值
    sortedDistIndicies = distances.argsort()
    # 定一个字典:统计类别次数
    classCount = {}

    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels_set[sortedDistIndicies[i]]
        # 统计类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # 把分类结果进行降序排序，然后返回得票数最多的分类结果
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 测试手写数字识别代码
def handWritingClassTest(k):
    # 创建测试集标签
    hwLabels = []
    # 加载训练数据
    trainingFileList = listdir('knn-digits/trainingDigits')
    # 获取文件夹下文件的个数
    m = len(trainingFileList)
    # 创建一个m行1024列的训练矩阵，该矩阵的每行数据存储一个图像
    trainingMat = np.zeros((m, 1024))
    # 从文件名中解析出分类数字，如文件8_15.txt的分类是8，它是数字8的第15个实例
    for i in range(m):
        # 获取文件名
        fileNameStr = trainingFileList[i]
        # 去掉 .txt
        fileStr = fileNameStr.split('.')[0]
        # 获取分类数字
        classNumStr = int(fileStr.split('_')[0])
        # 将获取到的分类数字添加到标签向量中
        hwLabels.append(classNumStr)
        # 将每一个文件的1x1024数据存储到训练矩阵中
        trainingMat[i, :] = img_to_vector("knn-digits/trainingDigits/%s" % fileNameStr)
        # 加载测试数据
    testFileList = listdir('knn-digits/testDigits')
    # 错误计数
    errorCount = 0.0
    # 测试数据的个数
    mTest = len(testFileList)
    # 从测试数据文件名中解析出分类数字
    for i in range(mTest):
        # 获取文件名
        fileNameStr = testFileList[i]
        # 去掉 .txt
        fileStr = fileNameStr.split('.')[0]
        # 获取分类数字
        classNumStr = int(fileStr.split('_')[0])
        # 获取测试集的1x1024向量,用于训练
        vectorTest = img_to_vector("knn-digits/testDigits/%s" % fileNameStr)
        # 返回分类结果
        result = classifier(vectorTest, trainingMat, hwLabels, k)
        print("the classifier came back with: %d, the real answer is: %d" % (result, classNumStr))
        if result != classNumStr:
            errorCount += 1.0
    # 输出错误个数
    print("\nthe total number of errors is: %d" % errorCount)
    # 输出错误率
    print("\nthe total error rate is: %f" % (errorCount / mTest))
    return errorCount


# 测试取不同的K值对于结果照成的影响的plot图
def selectK():
    x = list()
    y = list()
    max_i = 0
    for i in range(1, 5):
        x.append(int(i))
        y.append(int(handWritingClassTest(i)))
    plt.plot(x, y)
    plt.show()


# 开始测试，会生成折线图
selectK()
'''
k = 1  0.013742
k = 2  0.013742
k = 3  0.010571
k = 4  0.011628
'''
# 测试证明，k选3效果比较好
# handWritingClassTest(3)
