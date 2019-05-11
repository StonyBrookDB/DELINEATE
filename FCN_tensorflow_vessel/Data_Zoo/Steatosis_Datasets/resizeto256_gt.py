import os
import glob
import shutil
from PIL import Image

base_path = "new_region_boundary_gt/"
imList = os.listdir(base_path)

for file_name in imList:
	src_file_name = os.path.join(base_path, file_name)
	dest_file_name = os.path.join('new_region_boundary_gt_resized/', file_name)
	image = Image.open(src_file_name)
	img = image.resize((256,256), Image.ANTIALIAS)
	img.save(dest_file_name, quality=100)


