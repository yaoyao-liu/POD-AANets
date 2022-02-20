## POD-AANets

See the original GitHub repository here: <https://git.io/JYHyt>

We are still cleaning up the code for [PODNet](https://github.com/arthurdouillard/incremental_learning.pytorch) w/ AANets. 
If you need to use it now, here is a preliminary version.

### Installing requirements 
You need to install the same environment as [PODNet](https://github.com/arthurdouillard/incremental_learning.pytorch) to run this code.
You may see a list of the requirements [here](https://github.com/yaoyao-liu/POD-AANets/blob/main/requirements.txt).

### Preparing datasets

Our code uses exactly the same dataset splits as [PODNet](https://github.com/arthurdouillard/incremental_learning.pytorch).
You need to put the data of [ILSVRC2012] under this folder: <https://github.com/yaoyao-liu/POD-AANets/tree/main/data/imagenet>
The folders for the training and validation data should be named as `train` and `val`, respectively.

### Running experiments
You may run the experiments on ImageNet-Subset using the following command:
```bash
python run_exp.py
```

### Reporting issues
If you have any questions, please create an issue in [the original repository](https://git.io/JYHyt). Thanks!
