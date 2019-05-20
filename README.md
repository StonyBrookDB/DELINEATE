# DELINEATE
Deep Learning Based Steatosis Segmentation

## Introduction
CNN based model for delineating the boundaries of overlapped Steatosis droplets in whole-slide liver histopathology images. This project has three phases, first dil-Unet is used for steatosis region prediction and HNN model is used for boundary detection followed by the third model FCN-8s is used for integrating region and boundary information to generate the final prediction result. 

## Get this repo
```
git clone https://github.com/mousumi12/DELINEATE.git
```

### dil-Unet

**Details input/output**

- The input of the region prediction/dil-Unet model is RGB image containing multiple overlapped steatosis droplets(Figure left). The dil-Unet model produces pixel-wise classification having the same size of the original image. The value of each pixel corresponds to its class label (Figure right). There are two class labels where red corresponds to steatosis pixel and black corresponds to background/ non steatosis pixels.

Input image                          |  Region Prediction  
:-----------------------------------:|:----------------------------------:
<img src="https://github.com/mousumi12/DELINEATE/blob/master/Images/Original.png" width="200" height="180">  |  <img src="https://github.com/mousumi12/DELINEATE/blob/master/Images/Steat_region.png" width="200" height="180">

**Usage**

  Installing requirements
- Its recommended to install the requirements in a [conda virtual environment]. 
  ```
  pip install -r requirement_Unet_FCN.txt
  ```
  
**Seting up**

- The dataset folder is arranged according to the code structure requirement with one sample image inside each such folder
- The train/test data hierarchy should be organized in the following manner

```
    Use the Keras data generators to load train and test
    Image and label are in structure:
        train/
            img/
                0/
            gt/
                0/

        test/
            img/
                0/
            gt/
                0/

```

- Set up the necessary hpyerparameters 
- Launch training 

  ```
  cd dil-Unet
  python train.py --data_path ./datasets --checkpoint_path ./checkpoints/ --imSize 512
  ```
- We can visualize the train loss, dice score, learning rate, output mask, and first layer convolutional kernels per iteration in tensorboard

  ```
  tensorboard --logdir=train_log/
  ``` 
- When checkpoints are saved, we can use eval.py to test an input image with an arbitrary size.

- Launch evaluation (evaluate your model)
  ```
  python eval.py --data_path ./datasets/test/ --load_from_checkpoint ./checkpoints/model-7049 --batch_size 1 --imSize 512
  ```

### HNN

**Details input/output**

- The input of the HNN model is RGB image containing multiple overlapped steatosis droplets(Figure left). The HNN model produces the edge maps from side layers generated at 5k iterations. The final edge mapp is genarated as teh 5-th side output from HNN model (Figure right). 

Input image                          |    Boundary Detection  
:-----------------------------------:|:----------------------------------:
<img src="https://github.com/mousumi12/DELINEATE/blob/master/Images/Original.png" width="200" height="180">  |  <img src="https://github.com/mousumi12/DELINEATE/blob/master/Images/Steat_boundary.png" width="200" height="180">

**Usage**

Installing requirements
```
cd HNN/holy-edge
pip install -r requirements.txt
export OMP_NUM_THREADS=1
```

**Setting up**

- First step is to edit the [config file](https://github.com/mousumi12/DELINEATE/tree/master/HNN/holy-edge/hed/configs/hed.yaml) located at `hed/configs/hed.yaml`. 

Set the paths below. Make sure the directories exist and you have read/write permissions on them.
The HNN model is trained on (https://figshare.com/s/381f3c0200c87cae259e) dataset generated from Whole-slide images of liver tissue.

```
# location where training data : https://figshare.com/s/381f3c0200c87cae259e can be downloaded and decompressed
download_path: '<path>'
# location of snapshot and tensorbaord summary events
save_dir: '<path>'
# location where to put the generated edgemaps during testing
test_output: '<path>'
```

**VGG-16 base model**
VGG base model is available [here](https://github.com/machrisaa/tensorflow-vgg) used for producing multi-level features. The model is modified according with Section (3.) of the [paper](https://arxiv.org/pdf/1504.06375.pdf). Deconvolution layers are set with [tf.nn.conv2d_transpose](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose). The model uses single deconvolution layer in each side layers. Another implementation uses [stacked](https://github.com/ppwwyyxx/tensorpack/blob/master/examples/HED/hed.py#L35) bilinear deconvolution layers. 
The upsampling parameters are learned while finetuning of the model for this implementation. A pre-trained vgg16 net can be download from here[https://drive.google.com/file/d/0B6njwynsu2hXZWcwX0FKTGJKRWs/view?usp=sharing] or from here [ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy]


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

Region Pediction           |  Boundary Detection       |    Final Prediction
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/mousumi12/DELINEATE/blob/master/Images/Steat_region.png" width="200" height="180">  |  <img src="https://github.com/mousumi12/DELINEATE/blob/master/Images/Steat_boundary.png" width="200" height="180"> | <img src="https://github.com/mousumi12/DELINEATE/blob/master/Images/Steat_finalPred.png" width="200" height="180">

- The region and boundary predictions are combined and used to generate the final steatosis prediction using FCN-8s model. 
The net produces pixel-wise clasiification similar to the size of the image with the value of each pixel corresponding to its class (Figure right) where three classes corresponds to the backgorund, steatosis boundary and region pixel.

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

