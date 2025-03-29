# Introduction

This repository contains the source code of TU-Fold.

The datasets, models, and packages are available at [this release](https://github.com/ygjiyn/tu_fold/releases/tag/v0.0.1).

# Prerequisite

Our scripts use `Python` along with `torch` and `numpy`.

We think the scripts could run compatibly with various versions of those packages (`Python >= 3.9`, `torch >= 2.1`).

We list the versions we used for reference.

- Python
```
Python 3.11.9
```

- Packages
```
torch==2.4.1
numpy==1.26.4
```

# Usage

## Structure of the Dataset
The scripts assume the dataset has the following structure:

```
<DatasetName>
    <Subset1>
        train
        valid
        test
    <Subset2>
    ...
```

## Training and Evaluation

`exp_main.py` is used for training and evaluating the model.

For the example of each flag, please refer to the default value of each flag in the script.

When using the script, please consider setting the following flags if their default values are not your cases:
- `--dataset_root_path`: The path of `<DatasetName>` folder
- `--sub_dataset_list`: A string containing the names of `<Subset>` folders, separated by `:`
    - e.g., `subset_1:subset_2`
- `--result_path`: An existing path to place the results (e.g., the state dict and config file of the model)
- `--device`: `cuda` is chosen in default, set it to `cpu` if you do not have a `cuda` device

You could also adjust the hyper parameters of the model using other flags, whose names are self-evident.

## Prediction

After training, the `predict_all_samples.py` could be used for generating the prediction of samples for further analysis.

Please set the following flags
- `--results_folder_path`: the folder containing the state dict of the model `model_state_dict.pt` and its config file `config.json`. This could be set to the path used for `--result_path` in the training script described above
- `--save_path`: path of the folder for saving the predictions, the script will create this path
- `--sample_path_list`: a string containing the folders that include the samples to predict, separated by `:`

## Package

The `whl` package of our model could be installed using `pip`.

First, create a Python venv.

```
$ python3 -m venv tu_fold_venv
```

Activate this `venv`.

```
$ source tu_fold_venv/bin/activate
```

Install `tu_fold` in this `venv`.

```
$ (tu_fold_venv) python3 -m pip install tu_fold-0.0.1-py3-none-any.whl
```

Then `tu_fold` could be used as a command in the terminal

```
$ (tu_fold_venv) tu_fold --help
```

It has three parts, `train`, `predict_all`, and `pred`.

The flags of `train` and `predict_all` correspond to the `exp_main.py` and `predict_all_samples.py` scripts respectively, please refer to the description above.

As for `pred`, it is used to predict the structure of a single sequence, e.g.,

```
$ (tu_fold_venv) tu_fold pred AUCG...(the RNA sequence) --results_folder_path path/to/folder/containing/model/weight/and/config
```

This will use the corresponding model weight and config to predict the structure of the given sequence.

Quit the `venv`

```
$ (tu_fold_venv) deactivate
```

