# Learning Rewards from Linguistic Feedback

This repository contains (1) data, (2) model training, and (3) model analysis code.

## Requirements

Install the environment via Conda:
```
$ conda env create -f environment.yml
```

Run tests:
```
$ python -m unittest
```

## Data Exploration

Provided iPython notebooks under the `/notebooks` diectory can be used to explore datasets and re-run model evaluation.

To run them:
```
$ cd notebooks/
$ jupyter lab
```

### Datasets

The human-human and human-agent datasets can be explored with the `aaai_experiment_data_exploration.ipynb` notebook.

### Training

Training code / scripts are in the `aaai_inference_network_training.ipynb` notebook. 

### Evaluation

Evaluation code / scripts are in the `aaai_model_evaluation.ipynb.ipynb` notebook. 

These can be run independently of the training notebook.

### Pre-trained Models

Pretrained models are available in the `data/model_training_10fold` subdirectory. There is one `.pt` file for each cross-validation split. These models are loaded and used automatically in the `aaai_model_evaluation.ipynb` notebook.
