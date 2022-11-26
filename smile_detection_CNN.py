import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import numpy as np

file = open('datas', 'rb')
data = pickle.load(file)
file = open('labels', 'rb')
label = pickle.load(file)
#data_as = np.array(data)
#data = np.asarray(data).astype('float32')
#print(data)

le = LabelBinarizer()
label_bi = le.fit_transform(label)

x_train, x_test, y_train, y_test = train_test_split(data, label_bi, test_size=0.25)

aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode="nearest")

def network():
    net = tf.keras.models.Sequential([ 
                            tf.keras.layers.Conv2D(32,(3,3),  activation='relu', input_shape=(64,64,3)),
                            tf.keras.layers.BatchNormalization(),
                            tf.keras.layers.MaxPool2D(),

                            tf.keras.layers.Conv2D(64,(3,3),  activation='relu'),
                            tf.keras.layers.BatchNormalization(),
                            tf.keras.layers.MaxPool2D(),

                            tf.keras.layers.Flatten(),

                            tf.keras.layers.Dense(32, activation= 'relu'),
                            tf.keras.layers.BatchNormalization(),
                            tf.keras.layers.Dense(2, activation= 'softmax')     ])
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01, decay= 0.00025)
    net.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])
    return net
def show_results(N):    # Show Output
    plt.plot(N.history['accuracy'], label='train accuracy')
    plt.plot(N.history['val_accuracy'], label='test accuracy')
    plt.plot(N.history['loss'], label='train loss')
    plt.plot(N.history['val_loss'], label='test loss')
    plt.xlabel("epochs")
    plt.ylabel("accuracy/100")
    plt.legend()
    plt.title("smile detection classification")
    plt.show()

begin = time.time()
net = network()
print(net.summary())
print(type(x_train), x_train.shape)
'''N = net.fit(x_train, y_train, 
            validation_data = (x_test, y_test), 
            epochs = 15)
show_results(N)
net.save('smile_detection_cnn.h5')'''
end = time.time()
print("Total time of run for learning is {}".format(end-begin))
