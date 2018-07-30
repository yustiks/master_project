import  os
from  PIL import Image  as  im_open,  ImageDraw 
from keras.preprocessing import image as Image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model
import numpy as np

#model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('/scratch/ii1n17/model50_.h5')

files = os.listdir('/scratch/ii1n17/SD4060_patches/')
im = im_open.open('/lyceum/ii1n17/SD4060.TIF')
if not os.path.exists('/scratch/ii1n17/SD4060_patches_car/'):
    os.makedirs('/scratch/ii1n17/SD4060_patches_car/')
draw = ImageDraw.Draw(im)
k = 0
for el in files: 
    img_path = '/scratch/ii1n17/SD4060_patches/'+el
    img = Image.load_img(img_path, target_size=(224, 224))
    x = Image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
#    print(preds)
#    if (preds[0][0]>0.98):
    if (preds[0][0]>=0.9):
#        img.save('/scratch/ii1n17/SD4060_patches_car/'+el)
#        k += 1
# that is car = draw it
        str1 = el 
        str2 = '_'
        str3 = '.TIF'
        k2 = str1.find(str2)
        k3 = str1.find(str3)
        i = int(str1[0:k2])
        j = int(str1[k2+1:k3])
#        draw.rectangle([i, j, i+32, j+32], fill=128, outline=128)
        draw.line((i, j, i+32, j), fill="white", width=3)
        draw.line((i+32, j, i+32, j+32), fill="white", width=3)
        draw.line((i+32, j+32, i, j+32), fill="white", width=3)
        draw.line((i, j+32,i, j), fill="white", width=3)
        im.save('SD4060_resnet50.TIF')
