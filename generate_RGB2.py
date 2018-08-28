import numpy as np
from PIL import Image
import os 

files = os.listdir('no_car/')

if not os.path.exists('no_car1/'):
    os.makedirs('no_car1/')
k = 0
for el in files: 
    k+=1    
    im = Image.open('no_car/'+el, 'r')
    im.save('no_car1/'+str(k)+'.TIF')
print('number of cars is k=',k)
k+=1
R, G, B = 0, 1, 2
for i in range(1,k): 
    im1 = Image.open('no_car1/'+str(i)+'.TIF', 'r')
    # split the image into individual bands
    source = im1.split()
# process the green band
    out = source[G].point(lambda i: i * 0)
    source[G].paste(out, None, None)
    im = Image.merge(im1.mode, source)
    im.save('no_car1/'+str(k)+'.TIF')
    k +=1

    source = im1.split()
# process the green band
    out = source[G].point(lambda i: i * 0.5)
    source[G].paste(out, None, None)
    im = Image.merge(im1.mode, source)
    im.save('no_car1/'+str(k)+'.TIF')
    k +=1

    source = im1.split()
# process the green band
    out = source[R].point(lambda i: i * 0)
    source[R].paste(out, None, None)
    out = source[B].point(lambda i: i * 0)
    source[B].paste(out, None, None)
    im = Image.merge(im1.mode, source)
    im.save('no_car1/'+str(k)+'.TIF')
    k +=1

    source = im1.split()
# process the red band
    out = source[R].point(lambda i: i * 0)
    source[R].paste(out, None, None)
    im = Image.merge(im1.mode, source)
    im.save('no_car1/'+str(k)+'.TIF')
    k +=1

    source = im1.split()
# process the red band
    out = source[R].point(lambda i: i * 0.5)
    source[R].paste(out, None, None)
    im = Image.merge(im1.mode, source)
    im.save('no_car1/'+str(k)+'.TIF')
    k +=1

    source = im1.split()
# process the green band
    out = source[G].point(lambda i: i * 0)
    source[G].paste(out, None, None)
    out = source[B].point(lambda i: i * 0)
    source[B].paste(out, None, None)
    im = Image.merge(im1.mode, source)
    im.save('no_car1/'+str(k)+'.TIF')
    k +=1

    source = im1.split()
# process the blue band
    out = source[B].point(lambda i: i * 0)
    source[B].paste(out, None, None)
    im = Image.merge(im1.mode, source)
    im.save('no_car1/'+str(k)+'.TIF')
    k +=1

    source = im1.split()
# process the blue band
    out = source[B].point(lambda i: i * 0.5)
    source[B].paste(out, None, None)
    im = Image.merge(im1.mode, source)
    im.save('no_car1/'+str(k)+'.TIF')
    k +=1

    source = im1.split()
# process the green band
    out = source[R].point(lambda i: i * 0)
    source[R].paste(out, None, None)
    out = source[G].point(lambda i: i * 0)
    source[G].paste(out, None, None)
    im = Image.merge(im1.mode, source)
    im.save('no_car1/'+str(k)+'.TIF')
    k +=1

print('number is increased and now is ', k)
