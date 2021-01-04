# Proximal Policy Optimisation for generalisation in video games - code

Final project for the course Deep Learning (02456) at DTU.

- Stéphane G. - s192576
- Ástrós E. - s192590
- Simon W. N. - s153999

### 🎬Demo : https://www.youtube.com/watch?v=rD9wvL6qSqA&feature=youtu.be&ab_channel=St%C3%A9phaneGuichard

## Run a training job

- Open `jobscript.sh`
- Edit it with the hyperparameters wanted, and choose a run name.
- Some hyperparameters relative to procgen are only editable on `utils_train.py``
- Run the job with the following command on the DTU HPC: `bsub < jobscript.sh`
- When the job is running, a folder with the selected `run_name` will be created, with informations about the model progressively added to it.

It can also be run on any computer which works with cuda, by manually running the correct scripts.

➡️ When a model is training, it periodically register the training and evaluation reward. It also save the model weights, but only at the end of the training.

## Evaluate a model and generate a video

- Open `jobvalidation.sh` and edit the run name with the correct path to the trained model (this folder being the root).
- Edit validationV2.py : Import the model corresponding to the one you evaluate
The results will be save on a csv file, and a video will be generated. you can edit the name on the validationV2.py file

## Evaluate a model with multiple evaluations. No video

Same procedure as above, but with `jobvalidation_muli.sh`

## Generating plots

with `generate_****.py` files, one can generate plots with the model results. The plot used in the report have been generated with those scripts.

## Notes

**utils** are separated into `utils_test` and `utils_train` because we use different environments for training and testing. `training.py` imports `utils_train` and the validations scripts import `utils_test`.