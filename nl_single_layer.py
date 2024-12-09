
# example of a one-layer neural network (also known as a single-layer perceptron) in Python using NumPy
import numpy as np

# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

if __name__ == '__main__':
    # Input data (4 samples, 2 features each)
    inputs = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])

    # Expected output (4 samples, 1 output each)
    expected_output = np.array([[0],
                                [1],
                                [1],
                                [0]])

    # Initialize weights randomly with mean 0
    np.random.seed(1)
    weights = np.random.rand(2, 1)
    print(weights)
    # Learning rate
    learning_rate = 0.1
    # Number of iterations for training
    # iterations = 10000
    iterations = 10
    # Training the neural network
    for _ in range(iterations):
        # Forward pass
        input_layer = inputs
        outputs = sigmoid(np.dot(input_layer, weights))

        # Calculate the error
        error = expected_output - outputs
        print( error)
        # Backpropagation
        adjustments = error * sigmoid_derivative(outputs)
        weights += np.dot(input_layer.T, adjustments) * learning_rate

    # Print the final weights
    print("Trained weights:")
    print(weights)

    # Test the neural network with a new input
    new_input = np.array([1, 1])
    output = sigmoid(np.dot(new_input, weights))
    print("Output for new input [1, 1]:")
    print(output)