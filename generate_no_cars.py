from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image
import os 

#data augmentation
#we rotate every location window 11 times by: 0 ,
#4.5 , 9, · · · , 45 , then shrink or enlarge the non-rotating
#images into multi-scalings: 0.8,0.9,1.0,1.1,1.2,1.3.
# all the data is stored in the folder 'car'
import os
files = os.listdir('data_1004_resnet/no_car/')
# k = 294 previous data
k = 1000
for el in files: 
    im = Image.open('data_1004_resnet/no_car/'+el, 'r')
    #im.save('temp/0.TIF')
    k += 1
    im.rotate(90).save('data_1004_resnet/no_car/'+str(k)+'.TIF')
    k += 1
    im.rotate(90).transpose(Image.FLIP_LEFT_RIGHT).save('data_1004_resnet/no_car/'+str(k)+'.TIF')
    k += 1
    im.rotate(90).transpose(Image.FLIP_TOP_BOTTOM).save('data_1004_resnet/no_car/'+str(k)+'.TIF')
    k += 1
    im.rotate(180).save('data_1004_resnet/no_car/'+str(k)+'.TIF')
    k += 1
    im.rotate(270).save('data_1004_resnet/no_car/'+str(k)+'.TIF')
    k += 1
    im.transpose(Image.FLIP_LEFT_RIGHT).save('data_1004_resnet/no_car/'+str(k)+'.TIF')
    k += 1
    im.transpose(Image.FLIP_TOP_BOTTOM).save('data_1004_resnet/no_car/'+str(k)+'.TIF')


