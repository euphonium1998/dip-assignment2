import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread("Dataset/test/bird/ILSVRC2012_val_00001556.JPEG", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow('gra', gray)

    # gray = np.array(gray)  # np array
    # # 调节边缘亮度
    # gray[:25, :] = 0
    # gray[int(gray.shape[0]) - 25:int(gray.shape[0])] = 0
    # print('image shape: ', gray.shape)
    #
    # # step2 高斯去噪声
    # blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # cv2.imshow('blur', blurred)
    #
    # # step3 计算水平、垂直方向上的梯度
    # gradX = cv2.Sobel(gray, cv2.CV_32F, dx=1, dy=0)
    # gradY = cv2.Sobel(gray, cv2.CV_32F, dx=0, dy=1)
    # # x方向的梯度减去y方向上的梯度，留下具体高水平梯度和低垂直梯度的图像
    # gradient = cv2.subtract(gradX, gradY)
    # gradient = cv2.convertScaleAbs(gradient)
    # cv2.imshow('gradient', gradient)
    #
    # # step4 模糊二值化
    # _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    # # closed = cv2.dilate(thresh, None, iterations=4)
    # cv2.imshow('thr', thresh)
    #
    # # 找轮廓
    # cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # # 排序函数,找出最大连通域
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # print(len(cnts))
    #
    # # step5画轮廓
    # draw_img = cv2.drawContours(img, [cnts[0]], -1, (0, 0, 255), 3)
    # cv2.imshow('draw_img', draw_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
