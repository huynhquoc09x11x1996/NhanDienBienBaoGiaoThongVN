import os
import cv2
import numpy as np
from sklearn.externals import joblib

'''
    khu vuc xu ly opencv
'''

mSignCascade = cv2.CascadeClassifier("/Users/leclev1/Documents/LuanVanKHMT/PythonTSR/cascade3.xml")


def maybeRatioOfTS(ROI):
    rows = ROI.shape[0]
    cols = ROI.shape[1]
    if rows < (cols * (2 / 3)) or cols < (rows * (2 / 3)):
        return False
    return True


def detectAndDrawRectOnTrafficSignWithHaar(sock, img):
    padding = 5
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    signs = mSignCascade.detectMultiScale(gray
                                          , scaleFactor=1.1
                                          , minNeighbors=3
                                          , minSize=(24, 24))
    if len(signs) > 0:
        for (x, y, w, h) in signs:
            if w + padding < img.shape[0]:
                x -= padding
                y -= padding
                w += padding
                h += padding
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.imshow("ROI", img[y:(y + h), x:(x + w)])

            des_new = getDescriptorOneMat(img[y:(y + h), x:(x + w)])
            model = joblib.load("/Users/leclev1/Documents/LuanVanKHMT/PythonTSR/test_code_syntax/svm_model.pkl")
            print("predicted: " + str(model.predict(des_new)))
            print(img[y:(y + h), x:(x + w)].shape)

    cv2.imshow("CameraFromAndroidApp", img)


winSize = (64, 64)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
winStride = (8, 8)
padding = (8, 8)
locations = ((10, 20),)

# HOG feature
width = height = 48
hog = cv2.HOGDescriptor(winSize,
                        blockSize,
                        blockStride,
                        cellSize,
                        nbins,
                        derivAperture,
                        winSigma,
                        histogramNormType,
                        L2HysThreshold,
                        gammaCorrection,
                        nlevels)


def getDescriptorInFolder(list_object, class_name):
    descriptor = []
    for i, path in enumerate(list_object):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 128))
        hist = hog.compute(image, winStride, padding, locations)
        descriptor.append([np.asarray(hist).reshape(1, len(hist))[0], class_name])

    return descriptor


def getDescriptorOnePath(image_test_path):
    image = cv2.imread(image_test_path, cv2.IMREAD_GRAYSCALE)
    hist = hog.compute(image, winStride, padding, locations)
    return np.asarray(hist).reshape(1, len(hist))


def getDescriptorOneMat(mat):
    hist = hog.compute(mat, winStride, padding, locations)
    return np.asarray(hist).reshape(1, len(hist))


'''
    Khu vuc xu ly files
'''
static_path_data = '/Users/leclev1/Documents/LuanVanKHMT/PythonTSR/test_code_syntax/classes'
static_path_test = '../Test'


def getDanhSachFiles(static_path, f_n):
    return os.listdir(static_path + "/" + f_n)


def getListAbsolutePath_ofFolder(static_path, folder_name):
    list_path = []
    files = getDanhSachFiles(static_path, folder_name)
    for x in range(len(files)):
        if '.jpg' in files[x] or '.jpg' in files[x] or '.png' in files[x] or '.bmp' in files[x]:
            list_path.append(static_path + "/" + folder_name + "/" + files[x])
    return list_path
