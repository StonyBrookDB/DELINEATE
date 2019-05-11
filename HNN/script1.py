import os
import glob
import shutil

base_path = 'HED-BSDS/test'

'''
imList = os.listdir(base_path)
for i in imList:
    im = os.path.join(base_path,i)
    new_name = im.split('.')[0] + '.png'
    cmd1 = 'convert {} {}'.format(im, new_name)
    os.system(cmd1)
'''


color_files = glob.glob('HED-BSDS/test/*_color.png')

"""
label_files = [m.split('color.png')[0] + 'label.png' for m in color_files]
target_img = 'unet-tensorflow-keras/datasets/test/val/img/0/'
target_gt = 'unet-tensorflow-keras/datasets/test/val/gt/0/'
"""





for color_file, label_file in list(zip(color_files, label_files))[:10]:
	#shutil.copy(color_file, target_img)
        #shutil.copy(label_file, target_gt)
	target_imgname = os.path.join(target_img, os.path.basename(color_file))
	target_labelname = os.path.join(target_gt, os.path.basename(label_file))
        #print target_imgname
	#print target_labelname
	os.rename(color_file, target_imgname)
	os.rename(label_file, target_labelname)

