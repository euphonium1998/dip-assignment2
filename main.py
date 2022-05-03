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


def embeddingHOG(img):
    img = cv2.resize(img, (SIZE, SIZE))
    hog = cv2.HOGDescriptor()
    descriptors = hog.compute(img, winStride=(16, 16), padding=(0, 0))
    return descriptors

def embeddingHist(img):
    hist = cv2.calcHist([img], [0, 1, 2], None, [10, 10, 10], [0, 256, 0, 256, 0, 256])
    return hist.reshape(-1)

def imageEnhancement(img):
    imgList = []
    width, height, _ = img.shape
    img_width_box = int(width * 0.8)
    img_height_box = int(height * 0.8)
    for _ in range(10):
        start_pointX = int(random.uniform(0, width - img_width_box))
        start_pointY = int(random.uniform(0, height - img_height_box))
        copyImg = img[start_pointX:start_pointX + img_width_box, start_pointY:start_pointY + img_height_box]
        imgFlip = cv2.flip(copyImg, 1)
        imgList.append(copyImg)
        imgList.append(imgFlip)
    return imgList


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
                # gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
                feature = embeddingHOG(original)
                # feature = embeddingHist(original)
                node = [feature, categoryIdx]
                test.append(node)
        for root, dirs, files in os.walk(".\\Dataset\\train\\" + category, topdown=False):
            for name in files:
                path = os.path.join(root, name)
                original = cv2.imread(path, cv2.IMREAD_COLOR)
                imgList = imageEnhancement(original)
                for img in imgList:
                    # img = cv2.Canny(original, 128, 200)
                    # gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
                    feature = embeddingHOG(original)
                    # feature = embeddingHist(original)
                    node = [feature, categoryIdx]
                    train.append(node)
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
    print(trainX.shape)

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

    # img = cv2.imread("./Dataset/train/chicken/n01514668_728.JPEG", cv2.IMREAD_COLOR)
    # cv2.imshow('img', img)
    # print(img.shape)
    # imgList = imageEnhancement(img)
    # i = 0
    # for tmp in imgList:
    #     cv2.imshow(str(i), tmp)
    #     # cv2.imwrite(str(i), tmp)
    #     i = i + 1
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # print(gray.shape)
    # cv2.imshow('gra', gray)
    # img2 = cv2.flip(img, 1)
    # cv2.imshow('flip', img2)
    # resize = cv2.resize(img, (SIZE, SIZE))
    # print(resize.shape)
    # cv2.imshow('resize', resize)
    #
    # # cv2.drawKeypoints(image=img,
    # #                   outImage=img,
    # #                   keypoints=keypoints,
    # #                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    # #                   color=(255, 0, 255))
    # # cv2.imshow("SIFT", img)
    # hog = cv2.HOGDescriptor()
    # descriptors = hog.compute(resize, winStride=(8, 8), padding=(0, 0))
    # print(descriptors.shape)
    # hist = cv2.calcHist(img, [2], None,  [1000], [0, 256])
    # print(hist.shape)
    #
    cv2.waitKey(0)
    cv2.destroyAllWindows()
