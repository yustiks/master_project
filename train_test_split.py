from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image
import os 

import random

list1 = []
for i in range(1, 24001):
    list1.append(i)
random.shuffle(list1)

print(list1)

# 4800 test
# 3840 valid
# 15360 train 
files = os.listdir('big_data_1000/no_car/')
if not os.path.exists('big_data_1000/train/no_car/'):
    os.makedirs('big_data_1000/train/no_car/')
if not os.path.exists('big_data_1000/test/no_car/'):
    os.makedirs('big_data_1000/test/no_car/')
if not os.path.exists('big_data_1000/valid/no_car/'):
    os.makedirs('big_data_1000/valid/no_car/')

k = 0
m = 0
n = 0
for el in files: 
    k += 1
    el = list1[k-1]
    im = Image.open('big_data_1000/no_car/'+str(el)+'.TIF', 'r')
    if (k <=15360):
        im.save('big_data_1000/train/no_car/'+str(k)+'.TIF')
    elif (k >15360) and (k <=19200):
        m +=1
        im.save('big_data_1000/valid/no_car/'+str(m)+'.TIF')
    else: 
        n += 1
        im.save('big_data_1000/test/no_car/'+str(n)+'.TIF')
print(k)

# 4800 test
# 3840 valid
# 15360 train 
files = os.listdir('big_data_1000/car/')
if not os.path.exists('big_data_1000/train/car/'):
    os.makedirs('big_data_1000/train/car/')
if not os.path.exists('big_data_1000/test/car/'):
    os.makedirs('big_data_1000/test/car/')
if not os.path.exists('big_data_1000/valid/car/'):
    os.makedirs('big_data_1000/valid/car/')

k = 0
m = 0
n = 0
for el in files: 
    k += 1
    el = list1[k-1]
    im = Image.open('big_data_1000/car/'+str(el)+'.TIF', 'r')
    if (k <=15360):
        im.save('big_data_1000/train/car/'+str(k)+'.TIF')
    elif (k >15360) and (k <=19200):
        m +=1
        im.save('big_data_1000/valid/car/'+str(m)+'.TIF')
    else: 
        n += 1
        im.save('big_data_1000/test/car/'+str(n)+'.TIF')
print(k)
