import os
import numpy as np
import glob
from shutil import copyfile

color_files = glob.glob('img/*.png')

label_files = ["new_region_boundary_gt/" + os.path.basename(m) for m in color_files]
#label_files = ["new_region_boundary_gt/{}_RegionBoundary.png".format(os.path.basename(m).split('.png')[0]) for m in color_files]

train_img_dir = "Train_Image/"
train_label_dir = "Train_Label/"
val_img_dir = "Val_Image/"
val_label_dir = "Train_Label/"
test_img_dir = "Test_Image/"
test_label_dir = "Test_Label/"


directory = os.path.dirname(train_img_dir)
if not os.path.exists(directory):
       os.makedirs(directory)

directory = os.path.dirname(train_label_dir)
if not os.path.exists(directory):
       os.makedirs(directory)

directory = os.path.dirname(val_img_dir)
if not os.path.exists(directory):
       os.makedirs(directory)

directory = os.path.dirname(test_img_dir)
if not os.path.exists(directory):
	os.makedirs(directory)

directory = os.path.dirname(test_label_dir)
if not os.path.exists(directory):
       os.makedirs(directory)

for color_file, label_file in list(zip(color_files,label_files))[820:1025]:   #[0:588]: 

	target_imagename = os.path.join(val_img_dir, os.path.basename(color_file))
	target_labelname = os.path.join(val_label_dir, os.path.basename(label_file))

	copyfile(color_file, target_imagename)
	copyfile(label_file, target_labelname)


"""
color_files = glob.glob('Test_Image/*.png')
for color_file in color_files:
	target_file = os.path.join(test_img_dir, os.path.basename(color_file).split('_color.png')[0]+'.png')
        copyfile(color_file, target_file)
"""
