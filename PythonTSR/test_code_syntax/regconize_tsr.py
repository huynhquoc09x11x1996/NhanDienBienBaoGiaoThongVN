from myutils import *
from sklearn.externals import joblib

path_video_test = '/Users/leclev1/Documents/LuanVanKHMT/PythonTSR/video_test/IMG_0223.MOV'
cap = cv2.VideoCapture(path_video_test)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640,480))
    detectAndDrawRectOnTrafficSignWithHaar(None, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# img_test = cv2.imread(
#     "/Users/leclev1/Documents/LuanVanKHMT/PythonTSR/test_code_syntax/classes/019/019_1537278543935.jpg",
#     cv2.IMREAD_COLOR)
#
# des_new = getDescriptorOneMat(img_test)
# print(des_new)
# model = joblib.load("/Users/leclev1/Documents/LuanVanKHMT/PythonTSR/test_code_syntax/svm_model.pkl")
# print(model)
#
# print(model.predict(des_new))
#
# cv2.waitKey()
# cv2.destroyAllWindows()
