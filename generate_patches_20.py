# lets do no this sliding window just to see where in the picture is the car located
# sliding window will go through the whole piture and we will see if this particular square contains the car or not
from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image

#%matplotlib inline
im = Image.open('/lyceum/ii1n17/SD4060.TIF', 'r')
imshow(np.asarray(im))

# I will take picture half of which doesn`t even contain car (as it is sea )
# and I will check how many windows generated by sliding window contain car as an object 
size = 20
step = 10 
import os
if not os.path.exists('/scratch/ii1n17/patches_20/'):
    os.makedirs('/scratch/ii1n17/patches_20/')
width, height = im.size
for i in range(0,width-size,step):
    for j in range(0,height-size,step):
        area = (i, j, i+size, j+size)
        cropped_img = im.crop(area)
        cropped_img.save('/scratch/ii1n17/patches_20/'+str(i)+'_'+str(j)+'.TIF')
#        imshow(np.asarray(cropped_img))\n"