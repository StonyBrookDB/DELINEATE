# DELINEATE
Deep Learning Based Steatosis Segmentation

## Introduction
CNN based model for delineating the boundaries of overlapped Steatosis droplets in whole-slide liver histopathology images. This project has three phases, first dil-Unet is used for steatosis region prediction and HNN model is used for boundary detection followed by the third model FCN-8s is used for integrating region and boundary information to generate the final prediction result. 

### dil-Unet

**Usage**

Installing requirements
- Its recommended to install the requirements in a [conda virtual environment](https://conda.io/docs/using/envs.html#create-an-environment)
  ```
  pip install -r requirement_Unet_FCN.txt
  ```
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

**Seting up**

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
VGG base model available [here](https://github.com/machrisaa/tensorflow-vgg) is used for producing multi-level features. The model is modified according with Section (3.) of the [paper](https://arxiv.org/pdf/1504.06375.pdf). Deconvolution layers are set with [tf.nn.conv2d_transpose](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose). The model uses single deconvolution layer in each side layers. Another implementation uses [stacked](https://github.com/ppwwyyxx/tensorpack/blob/master/examples/HED/hed.py#L35) bilinear deconvolution layers. In this implementation the upsampling parameters are learned while finetuning of the model. Pre-trained weights for [VGG-16](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) are provided by [git-lfs](https://github.com/harsimrat-eyeem/holy-edge/blob/master/hed/models/vgg16.npy) in this repo.

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


