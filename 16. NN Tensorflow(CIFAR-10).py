import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Softmax, Dropout, Activation, BatchNormalization
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.utils import np_utils


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# plt.imshow(x_train[0])
# plt.show()
print(y_train)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# x_train /= 255              #Normalisation greatly improves performance. This Normalisation boosted acc from 65% to 71%
# x_test /= 255

mean = np.mean(x_train)               #This is Z-factor standardisation. Boosted acc from 65% to 76%
stddev = np.std(x_train)
x_train = (x_train - mean)/stddev
x_test =  (x_test - mean)/stddev



y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)



model = Sequential()
model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))
model.add(BatchNormalization())                                 #Applying Batch Noramlisation greatly improved performance in 10 epochs. 82% from 76%.
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))       #Without dropout layer and norm/standardisation. Training acc: 80%, Testing acc: 69%. Hence, it solves overfitting.
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add((MaxPooling2D(pool_size=(2,2))))
model.add(Dropout(0.3))
# model.add(Flatten())
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=64, epochs=125, verbose=1) #At 10 epochs, acc: 75%. After 125, acc:80%(W/o Batch Normalization).

#With Batch Normalisation, after 10 epochs, acc: 80%. After 125, acc: Training(94.6), Testing(84.5). (Data Overfitted)



score = model.evaluate(x_test, y_test, verbose=0)

print(score)

#Saving the weights and neural network to disk.
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')
