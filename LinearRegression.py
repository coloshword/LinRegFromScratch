## Linear Regression module from scratch 
import numpy as np

class LinearRegression():
    # will take optional parameters for number of epochs (passes through the data) and learning rate 
    def __init__(self, num_epochs=5, learning_rate=0.01):
        # weights matrix should be based on the number of features 
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weights = None # to be set in the fit function 
        self.bias = None # a scalar to be added to the prediction. 

    def fit(self, training_data, train_labels):
        # set weights and bias to be some random values 
        # since this is a linear function, both weights and biases will be 1 x m, where m is the number of features
        # the fact that it is 1 x m is actually important, because in linear regression, you only need one weight to control the influence of one feature 
        # everything making an output needs to have weights. Since we are only making one output in linear regression, we only need one set of weights for the system
        # versus in a neural network, each neuron in the NN is making an output, meaning each neuron needs a set of weights. Each neuron having a set of weights, means we need a vector of weights, which is a vector, meaning a weights matrix 
        # 1 x m is the dimensions of b and w 
        # training data therefore is m x n, where n is the number of training examples 
        # we will have rows be training examples, column = input features 
        # shape returns (rows, columns)
        # bias just shifts the values by a certain amount, so it needs only to be a scalar
        num_training_examples, num_features = np.shape(training_data)
        # initalize with random values, uses low, high, and size  which is 1 x num features 
        # keep it float32 because we don't need so much precision, but want speed
        self.weights = np.random.uniform(-1, 1, (num_features, 1)).astype(np.float32)
        # don't use astype, bacause bias is just one value ! 
        self.bias = np.float32(np.random.uniform(-1, 1))
        self.train(training_data, train_labels)

    def forward(self, x):
        return np.matmul(x, self.weights)  + self.bias

    def predict(self, x):
        return self.forward(x)

    def mean_squared_error(self, y, yhat):
        return np.mean((y - yhat) ** 2)

    def update(self, learning_rate, y, yhat, x):
        # batch gradient descent, we sum the gradients and then average it and update the values based on that 
        total_gradient = 0
        bias_gradient = 0

        # follow update rule, find the sum of the errors, which is (yi - yhat(i))*xj(i)
        # we don't need to do this for every theta j, because I am already providing y and yhat as vectors, and since I'm using np vector operations, I'm already accounting for all updates to parameters in theta
        # when i do (y - yhat) * x, I'm doing this to vectors meaning I'm doing this to every data point and for every parameter, at the same time, meaning each value in theta is getting it's own gradient contribution at the same
        # time
        for i in range(len(y)):
            y_at_idx = y[i]
            yhat_at_idx = yhat[i]
            x_at_idx = x[i]
            # sum the gradient
            total_gradient += (y_at_idx - yhat_at_idx) * x_at_idx
            # bias represents an offset, so it doesn't need to take into account the input value
            bias_gradient += (y_at_idx - yhat_at_idx)
        # once summed, update the value by multiplying the learning rate
        # reshape total_gradient
        total_gradient = total_gradient.reshape(self.weights.shape)
        self.weights += learning_rate * total_gradient / len(y)
        self.bias += learning_rate * bias_gradient / len(y)

    def train(self, train_inputs, train_labels):
        # train inputs n x m, where n is the number of examples, and m is the input features 
        print("Starting training")
        for epoch in range(self.num_epochs):
            print(f"Training on Epoch {epoch}")
            
            # Forward pass to predict
            yhat = self.forward(train_inputs)
            
            # Calculate loss (MSE as an example)
            loss = np.mean((train_labels - yhat) ** 2)  # Mean Squared Error
            
            # Print loss for this epoch
            print(f"Epoch {epoch} Loss (MSE): {loss}")
            
            # Update weights using gradient descent
            self.update(self.learning_rate, train_labels, yhat, train_inputs)


'''
implementation notes:

- for training, basically we will have a matrix m x n, where n is the number of training examples, and m is just the number of input features 
- the labels will be a 1 x n vector, where n is the number of training examples 

- fix update:
    - my current update is wrong, because we are only using it 
- train loop:
- loop for epoch:
    - we want to run update.
    - to run update we need y, and y hat.
    - we are doing batch gradient descent, so we are training on the entire dataset per epoch, hence train_labels is y.
    - yhat just means for each train_input we need to get the corresponding train_output 
    - ok so going back to getting yhat. we want a n x 1 size yhat 
    - forward with n x m input, gives you n x1 yhat

- figure out forward:
forward: x is inputs, which is of dimensions n x m. we want n x 1 
n x m  * m x 1, gives n x 1
- so that means weights is dim m x 1.
- matrix multiplication is np.matmul
'''