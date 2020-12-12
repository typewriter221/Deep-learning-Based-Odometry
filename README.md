# Deep-learning-Based-Odometry
## Prerequisites
- Python 3
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn
- Pandas
- SciPy
- numpy-quaternion
- tfquaternion
## Training

Datasetused [OxIOD](http://deepio.cs.ox.ac.uk/) 
## Pretrained models

Pretrained model is available in code folder named btp.hdf5
## Evaluation

1. Download the desired dataset and unzip it into the project folder (the path should be `"<project folder>/Oxford Inertial Odometry Dataset/handheld/data<id>/"`)
2. Run `python3 evaluate.py dataset model`, where `dataset` is either `oxiod` and `model` is the trained model file path (e.g. `6dofio_oxiod.hdf5`).
