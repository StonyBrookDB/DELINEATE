# DELINEATE
Deep Learning Based Steatosis Segmentation

## Introduction
CNN based model for delineating the boundaries of overlapped Steatosis droplets in whole-slide liver histopathology images. This project has three phases, first dil-Unet is used for steatosis region prediction and HNN model is used for boundary detection followed by the third model FCN-8s is used for integrating region and boundary information to generate the final prediction result. 

### dil-Unet
**Installing reqirement**

*Its recommended to install the requirements in a [conda virtual environment](https://conda.io/docs/using/envs.html#create-an-environment)
  ```
  pip install -r requirement_Unet_FCN.txt
  ```
* Check loader.py inside dil-Unet folder to organize the train/test data hierarchy 
* Set necessary hpyerparameters and run train.py 

  ```
  cd dil-Unet
  python train.py --data_path ./datasets --checkpoint_path ./checkpoints/ --imSize 512
  ```
* We can visualize the train loss, dice score, learning rate, output mask, and first layer convolutional kernels per iteration in tensorboard

  ```
  tensorboard --logdir=train_log/
  ``` 
- When checkpoints are saved, we can use eval.py to test an input image with an arbitrary size.

- Evaluate your model
  ```
  python eval.py --data_path ./datasets/test/ --load_from_checkpoint ./checkpoints/model-7049 --batch_size 1 --imSize 512
  ```

### HNN
**Installing requirements**

```
cd HNN/holy-edge
pip install -r requirements.txt
export OMP_NUM_THREADS=1
```

**Seting up**

* Edit the [config file](https://github.com/mousumi12/DELINEATE/tree/master/HNN/holy-edge/hed/configs/hed.yaml) located at `HNN/holy-edge/hed/configs/hed.yaml`. Set the paths below. Make sure the directories exist and you have read/write permissions on them.
The HNN model is trained on (http://vcl.ucsd.edu/hed/HED-BSDS.tar) set created by the authors.

```
# location where training data : http://vcl.ucsd.edu/hed/HED-BSDS.tar would be downloaded and decompressed
download_path: '<path>'
# location of snapshot and tensorbaord summary events
save_dir: '<path>'
# location where to put the generated edgemaps during testing
test_output: '<path>'
```

### FCN-8s
