# Learning Rewards from Linguistic Feedback

This repository contains (1) data, (2) model training, and (3) model analysis code to support the paper: https://arxiv.org/abs/2009.14715

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

`Appendix.pdf` contains additional information about models and experiments, including full transcripts from informative teacher-learner pairs. 

Provided iPython notebooks under the `/notebooks` diectory can be used to explore datasets and re-run model evaluation.

To run them:
```
$ cd notebooks/
$ jupyter lab
```

### Datasets

The human-human and human-agent datasets can be found in `notebooks/data/`: `human_trial_data.json` and `agent_trial_data.json` respectively.

The easiest way to get started with them is to use the `aaai_experiment_data_exploration.ipynb` notebook.

### Training

Training code / scripts are in the `aaai_inference_network_training.ipynb` notebook. The data augmentation step will cache results in the `notebooks/data/` subfolder.

### Evaluation

Evaluation code / scripts are in the `aaai_model_evaluation.ipynb` notebook. 

These can be run independently of the training notebook and will use pretrained models. Running it will cache results in the `notebooks/data/` subfolder.

### Pre-trained Models

Pretrained models are available in the `data/model_training_10fold` subdirectory. There is one `.pt` file for each cross-validation split. These models are loaded and used automatically in the `aaai_model_evaluation.ipynb` notebook.
