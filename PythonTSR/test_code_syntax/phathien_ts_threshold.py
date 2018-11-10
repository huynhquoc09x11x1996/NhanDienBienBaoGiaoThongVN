from myutils import *
import cv2

cap = cv2.VideoCapture('/Users/leclev1/Downloads/video2.mp4')
# cap = cv2.VideoCapture('/Users/leclev1/Downloads/315.MOV')

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    # origin = np.copy(frame)
    # red_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # blue_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #
    # # red range 1 (0-10)
    # lower_red = np.array([0, 50, 50])
    # upper_red = np.array([10, 255, 255])
    # r1_mask = cv2.inRange(red_frame, lower_red, upper_red)
    #
    # # red range 2 (170-180)
    # lower_red = np.array([170, 50, 50])
    # upper_red = np.array([180, 255, 255])
    # r2_mask = cv2.inRange(red_frame, lower_red, upper_red)
    #
    # # result sau khi split red
    # r_mask = cv2.bitwise_or(r1_mask, r2_mask)
    #
    # # blue range
    # lower_blue = np.array([100, 50, 50])
    # upper_blue = np.array([124, 255, 255])
    # b_mask = cv2.inRange(blue_frame, lower_blue, upper_blue)
    #
    # # mask sau khi megre tach do va tach xanh
    # final_mask = cv2.bitwise_or(b_mask, r_mask)
    # # blue de lam to nhieu~
    # # final_mask = cv2.blur(final_mask, (9, 9))
    #
    # # # phan nguong voi anh nhi phan
    # # ret, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
    #
    # # # phep an mon gian no
    # # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    # # final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    # # erode = cv2.erode(final_mask, None, iterations=4)
    # # final_mask = cv2.dilate(erode, None, iterations=4)
    #
    # _, cnts, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in cnts:
    #     # approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    #     area = cv2.contourArea(cnt)
    #     if 900.0 < area < 1500.0:
    #         x, y, w, h = cv2.boundingRect(cnt)
    #         if 0.8 * h < w < h or 0.9 * w < h < w:
    #             print("(h,w)", str(h) + "," + str(w))
    #             cv2.drawContours(origin, [cnt], -1, (0, 255, 0), 3)
    #             cv2.rectangle(origin, (x, y), (x + w, y + h), (0, 0, 255), 3)
    #
    # cv2.imshow("HSV", final_mask)
    # cv2.imshow("Original", origin)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
