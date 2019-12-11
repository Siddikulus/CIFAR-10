from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.preprocessing import image

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# for x in range(16):
#     plt.imshow(x_test[x])
#     plt.show()

mean = np.mean(x_train)               #This is Z-factor standardisation. Boosted acc from 65% to 74%
stddev = np.std(x_train)
x_train = (x_train - mean)/stddev
x_test =  (x_test - mean)/stddev

test_img = image.load_img('D:\\Grazziti\\Training_Python\\ML Algos\\CIFAR-10 TESTING IMAGES\\960x0.jpg', target_size=(32,32)) #Defining a random image.
test_img = image.img_to_array((test_img))
test_img = np.expand_dims(test_img, axis = 0)

# Load trained CNN model
json_file = open('D:\\Grazziti\\Training_Python\\ML Algos\\CIFAR-10_model(84.5%)\\CIFAR-10_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('D:\\Grazziti\\Training_Python\\ML Algos\\CIFAR-10_model(84.5%)\\CIFAR-10_weights.h5')

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
indices = np.argmax(model.predict(test_img),1)            #argmax returns index of maximum value of array.
for x in indices:
    print(labels[x])

