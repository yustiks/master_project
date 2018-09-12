from PIL import Image
import os

files = os.listdir('Images/001/')
for el in files:
    image_file = Image.open('Images/001/'+el)
    gray = image_file.convert('L')
#bw = gray.point(lambda x: 0 if x<128 else 255, '1')
    gray.save('001/'+el)
