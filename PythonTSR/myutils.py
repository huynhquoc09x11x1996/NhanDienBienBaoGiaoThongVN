import os
import cv2
import numpy as np
from sklearn.externals import joblib

'''
    khu vuc xu ly opencv
'''
dict_classes = {
    '001': 'cam nguoc chieu',
    '002': 'cam dau huyen',
    '003': '5T',
    '004': 'tam giac giao nhau 2 ben',
    '005': 'tam giac queo trai',
    '006': 'vong xoay xanh',
    '007': '2 nguoi qua duong',
    '008': 'vuong xanh 1 nguoi qua duong',
    '009': 'tam giac den giao thong',
    '010': 'cam oto queo trai',
    '011': 'dau cham thang',
    '012': 'vuong xanh quay dau',
    '013': 'tam giac silip',
    '014': 'tam giac duong doc cao',
    '015': '1 nguoi di bo cat ngang',
    '016': 'Khu vuc cho',
    '017': 'cam xe tho so',
    '018': 'tam giac chu Z nam ngang',
    '019': 'tron xanh mui ten dau huyen',
    '020': 'Di cham',
    '021': 'tam giac vang vong xoay',
    '022': 'Hang rao',
    '023': 'Sam set',
    '024': 'tam giac 2 duong song song',
    '025': 'Khoang cach 2 xe 2m',
    '026': 'toc do 20_30_40_50_60',
    '027': 'tron trang soc soc',
    '028': 'Stop',
    'unkown': 'unkown',
}

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
hog = cv2.HOGDescriptor(_winSize=(width, height),
                        _blockSize=(width // 2, height // 2),
                        _blockStride=(width // 4, height // 4),
                        _cellSize=(width // 2, height // 2),
                        _nbins=9,
                        _derivAperture=1,
                        _winSigma=-1,
                        _histogramNormType=0,
                        _L2HysThreshold=0.2,
                        _gammaCorrection=1,
                        _nlevels=64,
                        _signedGradient=True)


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
