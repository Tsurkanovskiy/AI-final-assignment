import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from keras.datasets import cifar10 
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Activation 
from keras.layers import Dropout 
from keras.layers.convolutional import Conv2D, MaxPooling2D 
from keras.utils import np_utils 
from tensorflow.keras.optimizers import SGD
from keras.models import load_model 
from PIL import Image

file_name = str(input("Введіть назву файлу: "))
im = Image.open(file_name)
img = []
pix = im.load()
for i in range(32):
	img.append([])
	for j in range(32):
		img[i].append(pix[i,j][0:3])

x = np.array([img])
x = x.astype('float32')
x /= 255 

model = load_model('my_model.h5')  
prediction = list(model.predict(x)[0])

decode_dict = {0:"літак", 1:"авто", 2:"птах", 3:"кіт", 4:"олень", 5:"собака", 6:"жаба", 7:"кінь", 8:"корабель", 9:"вантажівка"}
print(f"На малюнку {decode_dict[prediction.index(max(prediction))]}")