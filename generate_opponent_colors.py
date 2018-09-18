import os
from PIL import Image
import numpy as np
from shutil import copyfile

if not os.path.exists('004'):
    os.makedirs('004')

R,G,B = 0,1,2
k1 = 396
k2 = 791
k3 = 1186
for k in range(1, 396, 1):
    im1 = Image.open('003/'+str(k)+'.jpg')
    # take only blue + red
    source = im1.split()
    out = source[G].point(lambda i: i * 0)
    source[G].paste(out, None, None)
    im = Image.merge(im1.mode, source)
    im.save('004/'+str(k1)+'.jpg')
    copyfile('labels_3/'+str(k)+'.txt', 'labels_opponent/'+str(k1)+'.txt')
    k1 += 1
    
    im1 = Image.open('003/'+str(k)+'.jpg')
    # take only green + red
    source = im1.split()
    out = source[B].point(lambda i: i * 0)
    source[B].paste(out, None, None)
    im = Image.merge(im1.mode, source)
    im.save('004/'+str(k2)+'.jpg')
    copyfile('labels_3/'+str(k)+'.txt', 'labels_opponent/'+str(k2)+'.txt')
    k2 += 1

    im1 = Image.open('003/'+str(k)+'.jpg')
    # take only green + blue
    source = im1.split()
    out = source[R].point(lambda i: i * 0)
    source[R].paste(out, None, None)
    im = Image.merge(im1.mode, source)
    im.save('004/'+str(k3)+'.jpg')
    copyfile('labels_3/'+str(k)+'.txt', 'labels_opponent/'+str(k3)+'.txt')
    k3 += 1
