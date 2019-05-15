import os
import glob
from shutil import copyfile


color_files = glob.glob('img_copy/*.png')


idx = 0
for color_file in  color_files:
	idx = int(os.path.basename(color_file).split('.png')[0])
	target_imagename = "img/{0:04d}.png".format(idx)
        copyfile(color_file, target_imagename)
	idx = idx + 1

