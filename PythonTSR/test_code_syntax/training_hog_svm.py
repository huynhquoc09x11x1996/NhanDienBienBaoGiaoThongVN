from sklearn import svm
from myutils import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import sys

sys.path.append('/Users/leclev1/Documents/svm_opencv_files/libsvm-3.23/python')

# chuan bi hinh anh

print("Loading images in folder.....")
list_object_001 = getListAbsolutePath_ofFolder(static_path_data, '001')
list_object_002 = getListAbsolutePath_ofFolder(static_path_data, '002')
list_object_004 = getListAbsolutePath_ofFolder(static_path_data, '004')
list_object_005 = getListAbsolutePath_ofFolder(static_path_data, '005')
list_object_006 = getListAbsolutePath_ofFolder(static_path_data, '006')
list_object_007 = getListAbsolutePath_ofFolder(static_path_data, '007')
list_object_008 = getListAbsolutePath_ofFolder(static_path_data, '008')
list_object_010 = getListAbsolutePath_ofFolder(static_path_data, '010')
list_object_014 = getListAbsolutePath_ofFolder(static_path_data, '014')
list_object_015 = getListAbsolutePath_ofFolder(static_path_data, '015')
list_object_016 = getListAbsolutePath_ofFolder(static_path_data, '016')
list_object_017 = getListAbsolutePath_ofFolder(static_path_data, '017')
list_object_019 = getListAbsolutePath_ofFolder(static_path_data, '019')
list_object_unknown = getListAbsolutePath_ofFolder(static_path_data, 'unknown')
print("001: " + str(len(list_object_001)))
print("002: " + str(len(list_object_002)))
print("004: " + str(len(list_object_004)))
print("005: " + str(len(list_object_005)))
print("006: " + str(len(list_object_006)))
print("007: " + str(len(list_object_007)))
print("008: " + str(len(list_object_008)))
print("010: " + str(len(list_object_010)))
print("014: " + str(len(list_object_014)))
print("015: " + str(len(list_object_015)))
print("016: " + str(len(list_object_016)))
print("017: " + str(len(list_object_017)))
print("019: " + str(len(list_object_019)))
print("unkown: " + str(len(list_object_unknown)))
print("Finish loading images in folder..... ")

# descriptor [des,className]
print("Gettting descriptor...")
des001 = getDescriptorInFolder(list_object_001, '001')
des002 = getDescriptorInFolder(list_object_002, '002')
des004 = getDescriptorInFolder(list_object_004, '004')
des005 = getDescriptorInFolder(list_object_005, '005')
des006 = getDescriptorInFolder(list_object_006, '006')
des007 = getDescriptorInFolder(list_object_007, '007')
des008 = getDescriptorInFolder(list_object_008, '008')
des010 = getDescriptorInFolder(list_object_010, '010')
des014 = getDescriptorInFolder(list_object_014, '014')
des015 = getDescriptorInFolder(list_object_015, '015')
des016 = getDescriptorInFolder(list_object_016, '016')
des017 = getDescriptorInFolder(list_object_017, '017')
des019 = getDescriptorInFolder(list_object_019, '019')
desunkown = getDescriptorInFolder(list_object_unknown, 'unknown')
print("Finish gettting descriptor...")

# # all dataset
print("Prepare data for training...")
all_data = []
#
all_data.extend(des001)
all_data.extend(des002)
all_data.extend(des004)
all_data.extend(des005)
all_data.extend(des006)
all_data.extend(des007)
all_data.extend(des008)
all_data.extend(des010)
all_data.extend(des014)
all_data.extend(des015)
all_data.extend(des016)
all_data.extend(des017)
all_data.extend(des019)
all_data.extend(desunkown)
# # tach data va label
X = []
Y = []

for i, row_i in enumerate(all_data):
    X.append(row_i[0])
    Y.append(row_i[1])

OUTPUT = np.asarray(np.append(np.asmatrix(X), np.asmatrix(Y).T, axis=1))
df = pd.DataFrame(OUTPUT)
df.to_csv("data_6754.txt")

print("Finish preparing data for training!")
print("So luong sample: ", len(all_data))

data_train, data_test, label_train, label_test = train_test_split(X, Y, test_size=0.25, random_state=1)
clf = svm.LinearSVC(random_state=1)
clf.fit(data_train, label_train)
res = clf.predict(data_test)

count = 0
for i in range(len(res)):
    if res[i] == label_test[i]:
        count += 1

print("Do chinh xac: " + str((count * 100) / len(label_test)) + "%")

joblib.dump(clf, './svm_model.pkl')


