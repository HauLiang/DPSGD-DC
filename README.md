# DPSGD-DC

Python code for the paper "An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses".

If you use the code, please cite our paper:
> [Liang, Hao and Zhang, Wanrong and He, Xinlei and Wu, Kaishun and Xing, Hong, "An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses", *The 40th Annual AAAI Conference on Artificial Intelligence*, 2026.](https://openreview.net/forum?id=Zs20eeNxdb "https://openreview.net/forum?id=Zs20eeNxdb")



## A Fast Reproduction Guide

If you just want to reproduce the figures in our paper, navigate to the `Plot_figures` folder and run the corresponding plotting script (e.g., `plot_figure_x.py`). This will generate the x-th figure demonstrated in the paper.



## Preparation

- Dataset Download:

- Dataset Setup:



## Training

```shell
python train.py --train_dir ./data_split/One_train.txt --test_dir ./data_split/One_test.txt
```

After training, the experimental results will be saved in the `save_models` folder. 



@ All rights are reserved by the authors.
