<h1 align=center> Bone Fracture Detection Using X-rays

![](https://img.shields.io/badge/Python-3.9-blue) ![](https://img.shields.io/badge/torch-2.3.0-blue) ![](https://img.shields.io/badge/mlflow-2.13.0-blue) ![](https://img.shields.io/badge/flask-3.0.3-blue) ![](https://img.shields.io/badge/docker-7.1.0-blue) ![](https://img.shields.io/badge/Contributions-Welcome-brightgreen) ![](https://img.shields.io/badge/LICENSE-MIT-red)</h1>

<p align = left>Using a custom resnet50 model with a fine-tuned classification head to detect bone fractures from a patient's X-ray.</p>

## Overview

The model is capable of detecting bone fractures with a f1-score of 0.94 using just an X-ray from a patient.

## Hyperparameters

The main hyperparameters are number of training epochs, initial learning rate for our model, and gamma which is our learning rate decay parameter.

```python
epochs = 10
lr = 0.001
gamma = 0.99
```

The train script of our model has mlflow support to log these hyperparameters along with model performance measures like accuracy and loss on test data. To visit the mlflow ui just run the following in terminal.

```python
$ mlflow ui
```

<p align="center">
  <img src = assets/mlflow.png max-width = 50% height = '125' />
</p>

## How to train this model?

Just run the following script from your terminal

```python
$ python main.py
```
This will save the best version of your model in the "Models" directory.

And you'll see something like this on terminal. And yes the model is doing extremely well because I just used 200 images (instead of 9k images) to train for this example.

<p align="center">
  <img src = assets/train.png max-width = 50% height = '135' />
</p>

## How to run the flask app?

Just run the following script from your terminal

```python
$ flask --app app run
```
Go to the link displayed in terminal. You'll see something like this.

<p align="center">
  <img src = assets/app.png max-width = 100% height = '295' />
</p>

## How to run the docker container?

Just build the container using the following script on terminal. Replace my-app-name with a name that you'd like for your image.

```python
$ docker build -t my-app-name .
```
And once the image is built just run a container using

```python
$ docker run -d -p 5000:5000 my-app-name
```
## Model Performance

The model was trained on approximately 9k images, and evaluated on 1.3k images out of which the test data had around 500 images. The model's performance was really good during training and evaluation. Here is how the model did on training and validation data.

<p align="center">
  <img src = assets/model.png max-width = 80% height = '250' />
</p>

The best training and validation accuracies were close to 96% while the losses were below 0.15.

When evaluated on test data, the model had an accuracy close to 94% with a classification report as below.

<p align="center">
  <img src = assets/matrix.png max-width = 100% height = '300' />
</p>

## Dataset

The dataset can be downloaded from [here.](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project)