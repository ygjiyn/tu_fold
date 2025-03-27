# Introduction

This repository contains the source code of TU-Fold.

# Prerequisite

```
Python 3.11.9

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

# Scripts for Experiments

## Training and Evaluation

`exp_main.py` is used for training and evaluating the model.

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