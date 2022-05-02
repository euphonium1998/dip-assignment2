import cv2
import numpy as np
import random
import os
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

TOTAL_CATEGORY = 5
SIZE = 256


def embedding(img):

    hog = cv2.HOGDescriptor()
    descriptors = hog.compute(img, winStride=(16, 16), padding=(0, 0))
    return descriptors


def dataLoader():
    categoryList = ["dog", "chicken", "frog", "bird", "fish"]
    categoryIdx = 0
    train = []
    test = []
    for category in categoryList:
        for root, dirs, files in os.walk(".\\Dataset\\test\\" + category, topdown=False):
            for name in files:
                path = os.path.join(root, name)
                original = cv2.imread(path, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
                img = cv2.resize(gray, (SIZE, SIZE))
                feature = embedding(img)
                node = [feature, categoryIdx]
                test.append(node)
        for root, dirs, files in os.walk(".\\Dataset\\train\\" + category, topdown=False):
            for name in files:
                path = os.path.join(root, name)
                original = cv2.imread(path, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
                img = cv2.resize(gray, (SIZE, SIZE))
                feature = embedding(img)
                node = [feature, categoryIdx]
                train.append(node)
                # imgFlip = cv2.flip(img, 1)
                # feature = embedding(imgFlip)
                # node = [feature, categoryIdx]
                # train.append(node)
        categoryIdx = categoryIdx + 1
    return train, test


def shuffleInput(originalTrain, originalTest, size):
    random.shuffle(originalTrain)
    # random.shuffle(originalTest)
    trainX = np.zeros((len(originalTrain), size))
    trainY = np.zeros(len(originalTrain))
    testX = np.zeros((len(originalTest), size))
    testY = np.zeros(len(originalTest))
    for i in range(len(originalTrain)):
        trainX[i] = originalTrain[i][0]
        trainY[i] = originalTrain[i][1]
    for i in range(len(originalTest)):
        testX[i] = originalTest[i][0]
        testY[i] = originalTest[i][1]
    return trainX, trainY, testX, testY


if __name__ == '__main__':
    train, test = dataLoader()
    print(len(train[0][0]))
    trainX, trainY, testX, testY = shuffleInput(train, test, len(train[0][0]))

    # clf = svm.LinearSVC()
    # clf.fit(trainX, trainY)
    # predictY = clf.predict(testX)

    clf = SGDClassifier(loss="hinge", penalty="l2")
    clf.fit(trainX, trainY)
    predictY = clf.predict(testX)

    print(predictY)
    print(testY)

    tot = 0
    for i in range(len(predictY)):
        if predictY[i] == testY[i]:
            tot = tot + 1
    acc = tot / len(testY)
    print(acc)

    # img = cv2.imread("./Dataset/train/fish/n01443537_428.JPEG", cv2.IMREAD_COLOR)
    # print(img.shape)
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # print(gray.shape)
    # cv2.imshow('gra', gray)
    # img2 = cv2.flip(img, 1)
    # cv2.imshow('flip', img2)
    # resize = cv2.resize(img, (SIZE, SIZE))
    # print(resize.shape)
    # cv2.imshow('resize', resize)
    # sift = cv2.SIFT_create()
    # keypoints, descriptor = sift.detectAndCompute(gray, None)
    # descriptor = StandardScaler().fit_transform(descriptor)
    # print(descriptor.shape)
    # pca = PCA(n_components=100)
    # pca.fit(descriptor)
    # # print(pca.singular_values_)  # 查看特征值
    # print(pca.singular_values_.shape)
    # # print(pca.components_)  # 打印查看特征值对应的特征向量
    # print(pca.components_.shape)

    # cv2.drawKeypoints(image=img,
    #                   outImage=img,
    #                   keypoints=keypoints,
    #                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    #                   color=(255, 0, 255))
    # cv2.imshow("SIFT", img)
    # hog = cv2.HOGDescriptor()
    # descriptors = hog.compute(resize, winStride=(16, 16), padding=(0, 0))
    # print(descriptors.shape)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
