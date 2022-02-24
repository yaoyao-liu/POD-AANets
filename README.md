## POD-AANets

\[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Adaptive_Aggregation_Networks_for_Class-Incremental_Learning_CVPR_2021_paper.pdf)\] \[[Original GitHub Repository for AANets](https://git.io/JYHyt)\] 

See the original GitHub repository for AANet here: <https://git.io/JYHyt>

We are still cleaning up the code for [PODNet](https://github.com/arthurdouillard/incremental_learning.pytorch) w/ AANets. 
If you need to use it now, here is a preliminary version.

### Installing Requirements 
You need to install the same environment as [PODNet](https://github.com/arthurdouillard/incremental_learning.pytorch) to run this code.
You may see a list of the requirements [here](https://github.com/yaoyao-liu/POD-AANets/blob/main/requirements.txt).

### Preparing Datasets

Our code uses exactly the same dataset splits as [PODNet](https://github.com/arthurdouillard/incremental_learning.pytorch).

You need to put the data of [ILSVRC2012](https://www.image-net.org/) under this folder: <https://github.com/yaoyao-liu/POD-AANets/tree/main/data/imagenet>

The folders for the training and validation data should be named as `train` and `val`, respectively.

### Running Experiments
You may run the experiments on ImageNet-Subset using the following command:
```bash
python run_exp.py
```

### Results on ImageNet-Subset (5-phase)

You may see the log file [here](https://raw.githubusercontent.com/yaoyao-liu/POD-AANets/main/logs/imagenet_sub_5phase).
|       | Avg.  | Phase 0  | Phase 1 | Phase 2 | Phase 3  | Phase 4 | Phase 5 |
| -------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| `PODNet`            | 75.9 | 86.4 | 80.6 | 76.3 | 74.7 | 69.8 | 67.6 |
| `POD-AANets (paper)`  | 77.0 | 86.1 | 81.5 | 77.7 | 75.5 | 71.6 | 69.4 |
| `POD-AANets (this repo)`  | 77.3 | 87.0 | 82.2 | 78.5 | 75.5 | 71.6 | 69.0 |

### Reporting Issues
If you have any questions, please create an issue in [the original repository](https://git.io/JYHyt). Thanks!

### Citation

Please cite our paper if it is helpful to your work:

```bibtex
@inproceedings{Liu2020AANets,
  author    = {Liu, Yaoyao and Schiele, Bernt and Sun, Qianru},
  title     = {Adaptive Aggregation Networks for Class-Incremental Learning},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {2544-2553},
  year      = {2021}
}
```

### Acknowledgements

Our implementation uses the source code from the following repository:

* [PODNet: Pooled Outputs Distillation for Small-Tasks Incremental Learning](https://github.com/arthurdouillard/incremental_learning.pytorch)
