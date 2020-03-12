# UCFAdvancedComputerVision
My project work for UCF Advanced Computer Vision Spring 2020.

## Installation
Ensure that at least PyTorch 1.2.0 is installed and verify that `cuda/bin`, `cuda/include` and `cuda/lib64` are in your `$PATH`, `$CPATH` and `$LD_LIBRARY_PATH` respectively, *e.g.*:

```
$ python -c "import torch; print(torch.__version__)"
>>> 1.2.0

$ echo $PATH
>>> /usr/local/cuda/bin:...

$ echo $CPATH
>>> /usr/local/cuda/include:...
```
and
```
$ echo $LD_LIBRARY_PATH
>>> /usr/local/cuda/lib64
```
on Linux or
```
$ echo $DYLD_LIBRARY_PATH
>>> /usr/local/cuda/lib
```
on macOS.
Then run:

```sh
$ pip install --verbose --no-cache-dir torch-scatter
$ pip install --verbose --no-cache-dir torch-sparse
$ pip install --verbose --no-cache-dir torch-cluster
$ pip install --verbose --no-cache-dir torch-spline-conv (optional)
$ pip install torch-geometric
```

## Project 1
This project has three main objectives. 

1. Look at the affect of network size on performance by altering the number of convolutional layers. 
2. Look at the affect of training data size on performance by altering the size of samples per class. 
3. Observe transfer learning from a model trained on one dataset being fine-tuned on another. 

### Datasets
The dataset used for the first two objectives is Tiny ImageNet. This dataset has 100K image samples of 64x64x3 with 200 different
classes. The training task is classification.  

The dataset used for the third objective is Street View House Numbers (SVHN) dataset with images 32x32x3.

You can download the whole tiny ImageNet dataset from this link: ​ http://cs231n.stanford.edu/tiny-imagenet-200.zip
You can download the cropped images for SVHN from this link: ​ http://ufldl.stanford.edu/housenumbers/​

Under `utils/datasets.py` are two classes available to load the data. Make sure to change the SVHN `test_32x32.mat` to `val_32x32.mat` for the sake


### Running Experiments
There are two modeling approaches. The first creates a graphical representation feature output from a ResNet model and then uses [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric/blob/master/README.md) to perform a number of graphical convolutions until a final softmax classification. The second uses an AlexNet pre-trained on ImageNet and finetunes the final layer for the SVHN dataset.

Examples of how to run experiments for these models are found in `experiments.py`.

An example of calling the Finetuning experiments:

```python
from main import run_experiment
resize = (64, 64)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor(), normalize]) 
run_experiment('svhn', 'data/svhn', None, perc_class=perc_class, batch_size=128, epochs=50, num_classes=10, train=True, fine_tune=True, transform=transform)
```
An example of running the graphical model:

```python
num_conv = 4
run_experiment('imagenet', 'data/tiny-imagenet-200', num_conv, perc_class=100, batch_size=128, epochs=10, num_classes=200, train=True)
```