from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os 
from PIL import Image

image_file = Image.open('boat_aug/ship1.jpg')
gray = image_file.convert('L')
bw = gray.point(lambda x: 0 if x<128 else 255, '1')
bw.save('boat_aug/result.jpg')


