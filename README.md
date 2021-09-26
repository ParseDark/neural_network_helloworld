# Neural Network - MNIST Project 

A neural network hello world project. Just fun.

```python
> Dataset Source
> # https://pjreddie.com/projects/mnist-in-csv/
> # https://pjreddie.com/media/files/mnist_train.csv
> # https://pjreddie.com/media/files/mnist_test.csv
```

### How to run it on the local?
1. Clone the repo to your local
2. Create a fold and naming as "dataset"
3. Download Training Data from [mnist_train](https://pjreddie.com/media/files/mnist_train.csv) and put it in the dataset fold
4. Download Test Data from [mnist_test](https://pjreddie.com/media/files/mnist_test.csv) and put it in the dataset fold
5. Create a Virtual python enviroment(venv or conda: whether you want)
6. Install dependencies packages
    * numpy
    * scipy
    * matplotlib
7. Run The Command on virtual python environment(Step 6 created)
```python
python NeuralNetwork.py
```

### The network score
```python
{'correct': 9699, 'incorrect': 301, 'count': 10000}
```
