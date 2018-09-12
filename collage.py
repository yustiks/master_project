import sys
from PIL import Image
import os

images = map(Image.open, ['car1/1.TIF', 'car1/2.TIF', 'car1/3.TIF'])
 
total_width = 1950 #26 images x 75 pixels
max_height = 2925 # 75 pixels x 39 

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
y_offset = 0
files = os.listdir('car1/')
i = 0 
for el in files:
    if (i%26==0) and (i!=0):
        y_offset += 75
        x_offset = 0
    im = Image.open('car1/'+el, 'r')
    new_im.paste(im, (x_offset,y_offset))
    x_offset += 75
    i += 1
new_im.save('test1.jpg')
print(i)
