import socket
from myutils import *
import cv2
import numpy as np

TCP_IP = '192.168.43.1'
TCP_PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.connect((TCP_IP, TCP_PORT))
print("Connected to Android")

cv2.namedWindow("CameraFromAndroidApp", cv2.WINDOW_NORMAL)
cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)

imgData = ''

while True:
    k = cv2.waitKey(1)  # esc
    if k & 0xff is 27:
        sock.close()
        break
    data = sock.recv(1024)
    if not data:
        continue

    print(data)
    # imgData += data
    # a = imgData.find('\xff\xd8')
    # b = imgData.find('\xff\xd9')
    # if a != -1 and b != -1:
    #     feed = cv2.imdecode(np.fromstring(imgData[a:b + 2], dtype=np.uint8), 1)
    #     detectAndDrawRectOnTrafficSignWithHaar(sock, feed)
    #     imgData = imgData[b + 2:]

sock.close()
cv2.destroyAllWindows()
