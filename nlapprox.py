""" An example of using a neural network to perform nonlinear function approximation. The example shows a
    visualization of how the network converges into a solution
"""
import torch
import torch.nn.functional as nnfun
import matplotlib.pyplot as plt


def get_data_samples(num_samples, standard_dev):
    """ Generates noisy samples of a non-linear function y = f(x), where x and y are real values.
        The num_samples parameter indicates the number of required samples.
        Returns two column vectors for x and y represented as PyTorch tensors
    """

    # Prepare samples of a nonlinear function where x is defined in the interval [-1, 1]
    x_tensor = torch.linspace(-1, 1, num_samples)

    # x_tensor is a sequence of numbers, we need to convert it into a column vector
    x_tensor.resize_(len(x_tensor), 1)

    # Define a nonlinear function y = f(x) + n  where n is observation noise. Obtain samples of the
    # y variable also as a column vector
    y_tensor = x_tensor.tanh()*x_tensor.pow(4) + standard_dev*torch.rand(x_tensor.shape)

    return x_tensor, y_tensor


# In PyTorch, we define a neural network as a class that inherits from nn.Module. The class constructor
# defines the layers. We also need to implement a forward method that runs the neural network operations
# in the forward (normal) direction
class RegModel(torch.nn.Module):
    """ Defines a neural network with a single hidden layer
        Parameter num_vars represents the number of input variables
        Parameter num_hidden represents the number of hidden units
        Parameter num_output represents the number of output variables
    """
    def __init__(self, num_vars, num_hidden, num_output):
        super(RegModel, self).__init__()

        self.hidden = torch.nn.Linear(num_vars, num_hidden)
        self.predict = torch.nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = nnfun.relu(self.hidden(x))
        x = self.predict(x)
        return x


def train_and_visualize(model, num_iterations, x_tensor, y_tensor, optimizer, error_function):
    """ Trains a neural network using the defined model. Parameter num_iterations indicates the number of times we
        iterate over the data set. Parameters x_tensor and y_tensor are column vectors with the training data.
        The training uses optimization conditions defined in optimizer and minimizes the provided error function.
    """

    plt.ion()

    for ind in range(num_iterations):
        # predict the output of the model
        prediction = model(x_tensor)

        # compute the error between prediction and real output (in this case, we use the Mean Square Error)
        error = error_function(prediction, y_tensor)

        # set the gradients to zero
        optimizer.zero_grad()

        # Use the loss results and perform backpropagation; that is, update the model parameters based on
        # the gradient of the error with respect to parameters
        error.backward()

        # Update the optimizer for the next step
        optimizer.step()

        # display results every 5 iterations
        if ind % 5 == 0:
            # clear axis
            plt.cla()
            plt.title("Nonlinear function approximation of y = x^4 tanh(x) + n")
            plt.xlabel('x')
            plt.ylabel('y')
            # draw scatter plots of data
            plt.scatter(x_tensor.data.numpy(), y_tensor.data.numpy())

            # plot the approximated function using the model predictions
            plt.plot(x_tensor.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

            # display results for the current iteration
            display_text = "Iteration: {0}, MSE = {1:.4f}".format(ind, error.data.numpy())
            plt.text(-0.5, 0.8, display_text, fontdict={'size': 14, 'color': 'blue'})
            plt.pause(0.1)

    plt.ioff()
    plt.show()


if __name__ == "__main__":

    # Get 500 data samples of the nonlinear function. These samples are used to train a model. The data uses
    # the tensor representation of PyTorch
    x_ten, y_ten = get_data_samples(num_samples=500, standard_dev=0.2)

    # Define a neural network that has 1 input variable and 1 output variable. Also define
    # the number of hidden units
    nnmodel = RegModel(num_vars=1, num_hidden=20, num_output=1)

    # Define an optimizer function, in this case Stochastic Gradient Descent with a learning rate of 0.1
    optimizer_fun = torch.optim.SGD(nnmodel.parameters(), lr=0.3)

    # Define an error metric function, in this case we use the Mean Squared Error
    error_fun = torch.nn.MSELoss()

    # Train the neural network and visualze intermediate steps
    train_and_visualize(model=nnmodel, num_iterations=300, x_tensor=x_ten, y_tensor=y_ten,
                        optimizer=optimizer_fun, error_function=error_fun)
