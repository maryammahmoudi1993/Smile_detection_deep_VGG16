import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import glob
from mtcnn import MTCNN
import cv2
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pickle
import time

begin = time.time()
path = r"smile_dataset_600\\*\\*"

data = []
label = []
for i ,address in enumerate(glob.glob(path)):
    print(i,address)
    img_bgr = cv2.imread(address)
    
    img = cv2.resize(img_bgr, (64,64))
    #img_bgr
    #
    # cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(img.shape)

    detector = MTCNN()
    try:
        face_elements = detector.detect_faces(img)[0]
        #print(face_elements)
        x, y, w, h = face_elements['box']
        img = img[x:x+w, y:y+h]/255.0
        #print(img)
        data.append(img)
        label_name = address.split('\\')[-2]
        #print(label)
        label.append(label_name)
        #print('info: {} out of 2600 processed'.format(i))
        '''if i == 100:
            print('info: {} out of 3200 processed'.format(i))'''
        print(data.shape)
    except:
        pass
data = np.array(data, dtype=object)
#print(data)
'''lb = LabelBinarizer()
labels = lb.fit_transform(label)
labels = np.array(labels)'''
labels = np.array(label, dtype=object)
print(labels)

with open('datas', 'wb') as config_directory_file:
    pickle.dump(data, config_directory_file)
with open('labels', 'wb') as config_directory_file:
    pickle.dump(labels, config_directory_file)

end = time.time()
print("Total time of face detection for all dataset is {} second which is around {} minute".format(end-begin,(end-begin)/60))






