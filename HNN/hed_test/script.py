import os
import glob
import shutil
from PIL import Image

base_path = "model-49800-side5/"
imList = os.listdir(base_path)


for file_name in imList:
	src_file_name = os.path.join(base_path, file_name)
        #dest_file_name = os.path.join('model-49900-side5-resized/', file_name.split('testing-')[1])
	dest_file_name = os.path.join('model-49800-side5/', file_name.split('testing-')[1])
        #image = Image.open(src_file_name)
        #img = image.resize((256,256), Image.ANTIALIAS)
        #img.save(dest_file_name, quality=100)
        shutil.copyfile(src_file_name, dest_file_name)

