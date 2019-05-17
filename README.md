# DELINEATE
Deep Learning Based Steatosis Segmentation

## Introduction
CNN based model for delineating the boundaries of overlapped Steatosis droplets in whole-slide liver histopathology images. This project has three phases, first dil-Unet is used for steatosis region prediction and HNN model is used for boundary detection followed by the third model FCN-8s is used for integrating region and boundary information to generate the final prediction result. 

### dil-Unet

Input image                |  Region Prediction Image  
:-------------------------:|:-------------------------:
<img src="https://github.com/mousumi12/DELINEATE/blob/master/Images/Original.png" width="300">  |  <img src="https://github.com/mousumi12/DELINEATE/blob/master/Images/Steat_region.png" width="300">

**Usage**

  Installing requirements
- Its recommended to install the requirements in a [conda virtual environment]
  ```
  pip install -r requirement_Unet_FCN.txt
  ```
  
**Seting up**

- Check loader.py inside dil-Unet folder to organize the train/test data hierarchy 
- Set necessary hpyerparameters and run train.py 

  ```
  cd dil-Unet
  python train.py --data_path ./datasets --checkpoint_path ./checkpoints/ --imSize 512
  ```
- We can visualize the train loss, dice score, learning rate, output mask, and first layer convolutional kernels per iteration in tensorboard

  ```
  tensorboard --logdir=train_log/
  ``` 
- When checkpoints are saved, we can use eval.py to test an input image with an arbitrary size.

- Evaluate your model
  ```
  python eval.py --data_path ./datasets/test/ --load_from_checkpoint ./checkpoints/model-7049 --batch_size 1 --imSize 512
  ```

### HNN

**Usage**

Installing requirements
```
cd HNN/holy-edge
pip install -r requirements.txt
export OMP_NUM_THREADS=1
```

**Setting up**

- Edit the [config file](https://github.com/mousumi12/DELINEATE/tree/master/HNN/holy-edge/hed/configs/hed.yaml) located at `hed/configs/hed.yaml`. Set the paths below. Make sure the directories exist and you have read/write permissions on them.
The HNN model is trained on (http://vcl.ucsd.edu/hed/HED-BSDS.tar) set created by the authors.

```
# location where training data : http://vcl.ucsd.edu/hed/HED-BSDS.tar would be downloaded and decompressed
download_path: '<path>'
# location of snapshot and tensorbaord summary events
save_dir: '<path>'
# location where to put the generated edgemaps during testing
test_output: '<path>'
```

**VGG-16 base model**
VGG base model available [here](https://github.com/machrisaa/tensorflow-vgg) is used for producing multi-level features. The model is modified according with Section (3.) of the [paper](https://arxiv.org/pdf/1504.06375.pdf). Deconvolution layers are set with [tf.nn.conv2d_transpose](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose). The model uses single deconvolution layer in each side layers. Another implementation uses [stacked](https://github.com/ppwwyyxx/tensorpack/blob/master/examples/HED/hed.py#L35) bilinear deconvolution layers. In this implementation the upsampling parameters are learned while finetuning of the model. A pre-trained vgg16 net can be download from here[https://drive.google.com/file/d/0B6njwynsu2hXZWcwX0FKTGJKRWs/view?usp=sharing] or from here [ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy]


Pre-trained weights for [VGG-16](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) are provided by [git-lfs](https://github.com/harsimrat-eyeem/holy-edge/blob/master/hed/models/vgg16.npy) in this repo.

- Launch training
```
CUDA_VISIBLE_DEVICES=0 python run-hed.py --train --config-file hed/configs/hed.yaml
```
- Launch tensorboard
```
tensorboard --logdir=<save_dir>
```

- Launch Testing
Edit the snapshot you want to use for testing in `hed/configs/hed.yaml`

```
test_snapshot: <snapshot number>
```
```
CUDA_VISIBLE_DEVICES=1 python run-hed.py --test --config-file hed/configs/hed.yaml --gpu-limit 0.4
feh <test_output>
```
### FCN-8s

**Details input/output**

- The input for the net is RGB image (Figure 1 right).
The net produces pixel-wise annotation as a matrix in the size of the image with the value of each pixel corresponding to its class (Figure 1 left).

**Setup**

- Download a pre-trained vgg16 net and put in the /Model_Zoo subfolder in the main code folder. A pre-trained vgg16 net can be download from here[https://drive.google.com/file/d/0B6njwynsu2hXZWcwX0FKTGJKRWs/view?usp=sharing] or from here [ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy]

**Instructions for training (in TRAIN.py)**

- In: TRAIN.py
   1) Set folder of the training images in Train_Image_Dir
   2) Set folder for the ground truth labels in Train_Label_DIR
   3) The Label Maps should be saved as png image with the same name as the corresponding image and png ending
   4) Download a pretrained [vgg16](ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy) model and put in model_path (should be    done automatically if you have internet connection)
   5) Set number of classes/labels in NUM_CLASSES
   6) If you are interested in using validation set during training, set UseValidationSet=True and the validation image folder  to Valid_Image_Dir and set the folder with ground truth labels for the validation set in Valid_Label_Dir
  
**Instructions for predicting pixelwise annotation using trained net (in Inference.py)**

- In: Inference.py
   1) Make sure you have trained model in logs_dir (See Train.py for creating trained model)
   2) Set the Image_Dir to the folder where the input images for prediction are located.
   3) Set the number of classes in NUM_CLASSES
   4) Set  folder where you want the output annotated images to be saved to Pred_Dir
   5) Run script

**Evaluating net performance using intersection over union (IOU):**

-In: Evaluate_Net_IOU.py
  1) Make sure you have trained model in logs_dir (See Train.py for creating trained model)
  2) Set the Image_Dir to the folder where the input images for prediction are located
  3) Set folder for ground truth labels in Label_DIR. The Label Maps should be saved as png image with the same name as the corresponding image and png ending
  4) Set number of classes number in NUM_CLASSES
  5) Run script

