import os
from PIL import Image
files = os.listdir('car/')
for el in files:
    im = Image.open('car/'+el, 'r')
    im = im.resize((75,75))
    im.save('car1/'+el, im.format)
