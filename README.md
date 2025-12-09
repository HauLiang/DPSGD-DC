# DPSGD-DC

Python code for the paper "An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses".

If you use the code, please cite our paper:
> [[1] Liang, Hao and Zhang, Wanrong and He, Xinlei and Wu, Kaishun and Xing, Hong, "An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses", *The 40th Annual AAAI Conference on Artificial Intelligence*, 2026.](https://arxiv.org/abs/2502.17772 "https://arxiv.org/abs/2502.17772")



## A Fast Reproduction Guide

If you just want to reproduce the figures in our paper, navigate to the `Plot_figures` folder and run the corresponding plotting script (e.g., `plot_figure_x.py`). This will generate the x-th figure demonstrated in the paper.



## Preparation

- Environment:
    - Python==3.9.20, numpy==1.26.4, opacus==1.5.2
    - torch==2.5.0,  torchvision==0.20.1, scikit-learn==1.5.2


## Running Code

```shell
python attack.py 
```

By using this comment, the dataset will be saved in the `data` folder, and the experimental results will be saved in the `output` folder. 


##

This code is based on the code available from
https://github.com/csong27/membership-inference from the following paper:
> [[2] Shokri, Reza and Stronati, Marco and Song, Congzheng and Shmatikov, Vitaly. "Membership Inference Attacks Against Machine Learning Models". in *2017 IEEE symposium on security and privacy (SP)*, 2017.](http://ieeexplore.ieee.org/document/7958568 "http://ieeexplore.ieee.org/document/7958568")

Please check the accompanying license and the license of [2] before using. 


@ All rights are reserved by the authors.
