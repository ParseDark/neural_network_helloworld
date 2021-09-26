import numpy
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

# dataset
# https://pjreddie.com/projects/mnist-in-csv/
# https://pjreddie.com/media/files/mnist_train.csv
# https://pjreddie.com/media/files/mnist_test.csv

class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set num of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # initial the links weight
        # 创建权重矩阵
        # self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        # self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)

        # initial the sigmoid func
        self.activation_func = lambda x: special.expit(x)

        # initial the links weight
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        pass

    def train(self, input_list, target_list):
        # cover inputs list to 2d list
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_func(hidden_inputs)

        # # calculate signals into hidden layer
        # hidden_inputs2 = np.dot(self.wih, hidden_inputs)
        # # calculate the signals emerging from hidden layer
        # hidden_outputs2 = self.activation_func(hidden_inputs2)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_output = self.activation_func(final_inputs)

        # error is the (target - actual)
        output_errors = targets - final_output

        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_output * (1.0 - final_output)), np.transpose(hidden_outputs))
        
        # update the weight for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        

        pass

    def query(self, input_list):
        # cover inputs list to 2d list
        inputs = np.array(input_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_func(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_output = self.activation_func(final_inputs)

        return final_output

def showTheMnist():
    data_file = open("./dataset/mnist_train.csv", "r")
    data_list = data_file.readlines()
    data_list
    data_file.close()
    print(len(data_list))
    all_values = data_list[2].split(',')
    image_arr = np.asfarray(all_values[1:]).reshape([28, 28])
    plt.imshow(image_arr, cmap='Greys', interpolation='None')


def TestNetwork():
    # why 784? because 28 * 28 equal 784
    input_nodes = 784
    # 判断隐藏层的节点数需要多次的实验
    hidden_nodes = 200
    output_nodes = 10
    # 学习率
    learningrate = 0.2
    n = NeuralNetwork(inputnodes=input_nodes, hiddennodes=hidden_nodes, outputnodes=output_nodes, learningrate=learningrate)
    # load training data from dataset
    data_file = open("./dataset/mnist_train.csv", "r")
    data_list = data_file.readlines()
    data_file.close()

    # train the neural network
    # go throught all records in the traning dataset
    epochs = 7

    for e in range(epochs):
        for record in data_list:
            # split the record by the ', '
            all_values = record.split(',')
            # scale and shift the inputs
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output value (all 0.01, except the desired label which is 0.99)
            targets = np.zeros(output_nodes) + 0.01
            # all_values[0 is the target label for this record
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
        print(f"completed epoch{e}")

    return n


def getTestDataset():
    data_file = open("./dataset/mnist_test.csv", "r")
    data_list = data_file.readlines()
    data_file.close()
    return data_list;


def singleTest(x, n, c):
    all_values = x.split(',')
    actural_result = all_values[0]
    formatInput = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    predict_result_percent = n.query(formatInput)
    predict_result = predict_result_percent.argmax()
    isCorrect =int(actural_result) == int(predict_result)
    # print(f"actural result {(actural_result)}----predict result {(predict_result)}----{isCorrect}")
    c['count'] += 1
    if isCorrect:
        c['correct'] += 1
    else:
        c['incorrect'] += 1


def RunTest():
    dataset = getTestDataset()
    n = TestNetwork()
    c = {
        "correct": 0,
        "incorrect": 0,
        "count": 0
    }

    [singleTest(x, n, c) for x in dataset]
    print(c)
    print(f"the network score is{c.get('correct') / c.get('count')}")


if __name__ == '__main__':
    RunTest()



