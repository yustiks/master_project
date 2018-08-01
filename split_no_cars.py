from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image
import os 


import os
files = os.listdir('data_1004_resnet/car/')
k = 0
for el in files: 
    k += 1
    im = Image.open('data_1004_resnet/car/'+el, 'r')
    im.save('big_data_1000/car/'+str(k)+'.TIF')
print(k)

files = os.listdir('data_1004_resnet/no_car')
k = 0
for el in files: 
    k += 1
    im = Image.open('data_1004_resnet/no_car/'+el, 'r')
    im.save('big_data_1000/no_car/'+str(k)+'.TIF')
print(k)
