# MECIL
[Memory-Efficient Class-Incremental Learning for Image Classification](https://ieeexplore.ieee.org/abstract/document/9422177/), TNNLS 2021.

## Requirements
- Theano (version 0.9.0)
- Lasagne (version 0.2.dev1)
- Numpy (working with 1.11.1)
- Scipy (working with 0.18)
- CIFAR 100 downloaded 

## How to perform training
Execute ``main_cifar_100_theano.py`` to launch the training code. Settings can easily be changed by hardcoding them in the parameters section of the code. ``eval_cifar.py`` evaluates the performances of a trained network on different groups of classes.

PS: before running the main file, the path to the data location can be changed in ``utils_cifar100.py``

## Thanks
Our code is modified from [iCaRL](https://github.com/srebuffi/iCaRL/tree/master/iCaRL-TheanoLasagne).

## Citation
```BibTeX
@article{zhao2021memory,
  title={Memory efficient class-incremental learning for image classification},
  author={Zhao, Hanbin and Wang, Hui and Fu, Yongjian and Wu, Fei and Li, Xi},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2021},
  publisher={IEEE}
}
```
