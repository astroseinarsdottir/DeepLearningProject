# DeepLearningProject

Final project for the course Deep Learning

- Stéphane
- Astros
- Simon

## Some things to know

- The **model** used for **training** is on `model.py`
- The model on `model_test.py` can be modified and used to **evaluate** with another model architecture than the main one.
- `validationV2.py` put a result on `validation.csv`(or create the file if doesn't exist)
- `validation_multi.py` runs validation on 5 differents seeds, and save the results.
- ⚠️ When validating, dont forget to change the file names for the graph, or it will overwrite it.

**utils** are separated into `utils_test` and `utils_train` because we use different environments for training and testing. `training.py` imports `utils_train` and the validations scripts import `utils_test`.

## How to run on HPC

`bsub < jobscript.sh`
`bstat`
