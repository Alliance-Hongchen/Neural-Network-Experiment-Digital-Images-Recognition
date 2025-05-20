import numpy as np
from numpy import random
import os
import cv2 as cv
import time
import matplotlib.pyplot as plt


# 第一部分：数据处理
# 将任意像素数量图像压缩成5*5的像素大小，并根据其灰度值转换为01矩阵储存在列表中
def load_images(path):
    images = []
    for file in os.listdir(path):

        def showImage(img):
            cv.namedWindow('img', 0)
            cv.resizeWindow('img', 200, 200)
            cv.imshow("img", img)
            cv.waitKey()

        img = cv.imread(os.path.join(path, file), 1)
        #  showImage(img)

        #  print('原图的shape', img.size)

        def changeImage(img):
            size = (5, 5)
            img_new = cv.resize(img, size, interpolation=cv.INTER_AREA)
            return img_new

        img1 = changeImage(img)
        #  print(img1)

        matrix = np.zeros((5, 5)).astype(int)

        for i in range(5):
            for j in range(5):
                if img1[i][j][0] < 128:
                    matrix[i][j] = 1
        #  print(matrix)

        #  showImage(img1)
        #  print('缩放后的shape', changeImage(img).size)

        images.append(matrix)

    return np.array(images)


# 考虑到训练过程中输入向量为25*1的列向量，该函数用于将灰度值矩阵转换为列向量用作输入
def change_to_lie(x):
    res = []
    for i in range(5):
        for j in range(5):
            res.append(x[i][j])

    return res


# 数据处理具体实现，训练集和测试集输入向量分别存入train_in和test_in列表中
train_images = load_images('train_p')
test_images = load_images('test_p')

train_in = []
for i in range(5):
    train_in.append(change_to_lie(train_images[i]))

test_in = []
for i in range(5):
    test_in.append(change_to_lie(test_images[i]))

train_in = np.array(train_in)
test_in = np.array(test_in)

# 初始化训练集正确输出
D = np.mat([[1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]])


# 第二部分：初始化
# 定义神经网络结构
input_size = 25  # 输入层节点数
print("单隐层模型，请输入隐藏层节点数：")
hidden_size = int(input())  # 用户自定义隐藏层节点数
output_size = 5  # 输出层节点数

print("请选择激活函数类型：1为Sigmoid；2为ReLu")
activation = int(input())  # 用户自定义激活函数类型

print("请输入学习率：")
alpha = float(input())  # 用户自定义学习率

print("请输入代价函数选择：1为平方损失代价函数；2为交叉熵代价函数")
cost_function = int(input())  # 用户自定义代价函数类型

print("请输入隐藏节点dropout率：")
drop_rate = float(input())  # 用户自定义dropout率

# 权值矩阵初始化，矩阵元素值为-1~1
W1 = 2 * random.random(size=(hidden_size, input_size)) - 1
W2 = 2 * random.random(size=(output_size, hidden_size)) - 1


# 第三部分：函数

# 精度限制函数
# 在python中，小数位数很大的两个数相乘可能会导致结果为NaN，故需要定期对相关参数进行精度调整
# 经过实际测试，若每次训练结束后均对权值矩阵W进行精度调整，1000次训练大概耗时为30s
# 为了提升效率，我设置每10次训练结束后进行精度限制，时间效率得以显著提升，1000次训练大概耗时为2s
def change_jingdu_w(x1, x2):
    res1 = []
    res2 = []
    x1 = np.mat(x1)
    x2 = np.mat(x2)

    for i in range(hidden_size):
        b = x1.getA()[i]
        c = []
        for j in range(25):
            c.append(round(b[j], 6))
        res1.append(c)

    for i in range(5):
        b = x2.getA()[i]
        c = []
        for j in range(hidden_size):
            c.append(round(b[j], 6))
        res2.append(c)

    res1 = np.array(res1)
    res2 = np.array(res2)

    return res1, res2


# 隐藏层Sigmoid激活函数
def sigmoid(x):
    x = np.mat(x)
    x = x.T

    b = x.getA()[0]
    res = []

    for i in range(hidden_size):
        c = b[i]
        f = 1 / (1 + np.exp(-c))
        res.append(f)

    res = np.array(res)

    return res.T


# 隐藏层ReLU激活函数
def relu(x):
    x = np.mat(x)
    x = x.T

    b = x.getA()[0]
    res = []

    for i in range(hidden_size):
        res.append(max(0, b[i]))

    res = np.array(res)

    return res.T


# 输出层Softmax输出函数
def softmax(x):
    x = np.mat(x)
    x = x.T
    b = x.getA()[0]

    e = []
    sum1 = 0

    for i in range(5):
        sum1 += np.exp(b[i])

    for i in range(5):
        e.append(np.exp(b[i])/sum1)

    return np.array(e)


# 节点丢失Dropout函数
def dropout(x):
    x_len = len(x)

    for i in range(0, x_len):
        if random.random() < drop_rate:
            x[i] = 0

    res = np.mat(x)

    return res.T


# 反向训练误差计算函数
# get_delta和relu_delta两个函数用于计算反向传播的误差的delta值
def get_delta(y, e, check):
    res = []
    a1 = y.T
    a2 = e.T
    b1 = a1.getA()[0]
    b2 = a2.getA()[0]

    if check == 0:
        for i in range(5):
            res.append(b1[i] * (1 - b1[i]) * b2[i])
    elif check == 1:
        for i in range(hidden_size):
            res.append(b1[i] * (1 - b1[i]) * b2[i])

    res = np.mat(res)

    return res.T


def relu_delta(v, e):
    res = []
    a1 = v.T
    a2 = e.T
    b1 = a1.getA()[0]
    b2 = a2.getA()[0]

    for i in range(len(b1)):
        if b1[i] > 0:
            res.append(b2[i])
        else:
            res.append(0.0)

    res = np.mat(res)

    return res.T


# 训练函数（误差函数分别为交叉熵和均方）
def MuiltiClass_ce(W1, W2, X, D):
    cost = 0
    for i in range(5):
        x = X[i]
        x = np.mat(x)  # 变为矩阵形式
        x = x.T  # 列向量
        d = D[i].T

        v1 = np.dot(W1, x)
        if activation == 1:
            y1 = sigmoid(v1)
        else:
            y1 = relu(v1)

        y1 = dropout(y1)

        v = np.dot(W2, y1)
        y = softmax(v)

        y = np.mat(y)
        y = y.T

        e = d - y
        delta = e
        cost += np.sum(np.square(e))  # 计算误差的平方和

        e1 = np.dot(W2.T, delta)
        if activation == 1:
            delta1 = get_delta(y1, e1, 1)
        else:
            delta1 = relu_delta(v1, e1)

        dW1 = alpha * np.dot(delta1, x.T)
        W1 = W1 + dW1

        y1 = np.mat(y1)  # 必须这一步，否则下方矩阵乘法报错

        dW2 = alpha * np.dot(delta, y1.T)
        W2 = W2 + dW2

    cost_history.append(cost)
    return W1, W2


def MuiltiClass_mse(W1, W2, X, D):
    cost = 0
    for i in range(5):
        x = X[i]
        x = np.mat(x)  # 变为矩阵形式
        x = x.T  # 列向量
        d = D[i].T

        v1 = np.dot(W1, x)
        if activation == 1:
            y1 = sigmoid(v1)
        else:
            y1 = relu(v1)

        y1 = dropout(y1)

        v = np.dot(W2, y1)
        y = softmax(v)

        y = np.mat(y)
        y = y.T

        e = d - y
        delta = get_delta(y, e, 0)
        cost += np.sum(np.square(e))  # 计算误差的平方和

        e1 = np.dot(W2.T, delta)
        if activation == 1:
            delta1 = get_delta(y1, e1, 1)
        else:
            delta1 = relu_delta(v1, e1)

        dW1 = alpha * np.dot(delta1, x.T)
        W1 = W1 + dW1

        dW2 = alpha * np.dot(delta, y1.T)
        W2 = W2 + dW2

    cost_history.append(cost)
    return W1, W2


# 测试集检测函数check_test_p
def check_test_p(X):
    res = []
    for i in range(5):
        x = X[i]
        x = np.mat(x)  # 变为矩阵形式
        x = x.T  # 列向量

        v1 = np.dot(W1, x)
        if activation == 1:
            y1 = sigmoid(v1)
        else:
            y1 = relu(v1)
        v = np.dot(W2, y1)

        v = np.mat(v)
        v = v.T

        y = softmax(v)

        res.append(y)

    return res


start = time.time()  # 记录训练时间
cost_history = []  # 记录误差值，用于绘制误差下降曲线

# 这里的cost_function为用户自定义的误差函数类型，1为mse，2为ce
if cost_function == 1:
    for i in range(10000):  # 训练次数可手动调整
        W1, W2 = MuiltiClass_mse(W1, W2, train_in, D)
        if i % 100 == 0:  # 调整精度频次可参考：训练次数/100，进行计算（即保证整个训练过程平均调整100次精度）
            W1, W2 = change_jingdu_w(W1, W2)

    # 记录训练结束时间，并输出训练时间
    end = time.time()
    print("训练时间为：%.6fs" % (end - start))

    # 获得测试集测试结果，并输出
    ans = check_test_p(test_in)

    for i in range(5):
        print("分析第%d张图片:" % (i + 1))
        a = ans[i]

        for j in range(5):
            print("该图像为数字%d的概率为：%.6f" % (j + 1, a[j]))

elif cost_function == 2:
    for i in range(10000):  # 训练次数可手动调整
        W1, W2 = MuiltiClass_ce(W1, W2, train_in, D)
        if i % 100 == 0:  # 调整精度频次可参考：训练次数/100，进行计算（即保证整个训练过程平均调整100次精度）
            W1, W2 = change_jingdu_w(W1, W2)

    # 记录训练结束时间，并输出训练时间
    end = time.time()
    print("训练时间为：%.6fs" % (end - start))

    # 获得测试集测试结果，并输出
    ans = check_test_p(test_in)

    for i in range(5):
        print("分析第%d张图片:" % (i + 1))
        a = ans[i]

        for j in range(5):
            print("该图像为数字%d的概率为：%.6f" % (j + 1, a[j]))


# 误差下降曲线绘制
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
str = 'Error Decrease '
if activation == 1:
    str += 'by Sigmoid '
else:
    str += 'by Relu'
str += f" with alpha = {alpha}"
plt.title(str)
plt.show()
