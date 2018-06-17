from numpy import random, dot, exp, array
import matplotlib.pyplot as plt

# 31 because it has 3 inputs and 1 output
class NeuralNetwork():

	def __init__(self):

		# fixing the random seed for reproducibility
		random.seed(1)

		# inizializing uniform random weights in the interval [-1, 1]
		self.synaptic_weights = 2 * random.random((3,1)) - 1
		self.error_history = []


	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))


	def __sigmoid_derivative(self, x):
		return x * (1 - x)


	def train(self, given_input, given_output, epochs):

		self.error_history.clear()

		# train cycle
		for i in range(epochs):

			# one forward pass of all the inputs. Batch processing.
			output = self.forward(given_input)

			# the difference is the error
			error = output - given_output 
			self.error_history.append(error.mean())

			# calculating the gradient
			adjustement = dot(given_input.T, error * self.__sigmoid_derivative(output))

			self.synaptic_weights += adjustement


	def forward(self, inputs):
		return self.__sigmoid(dot(inputs, self.synaptic_weights))




def main():

	# inizialiting the neural network
	neural_network = NeuralNetwork()


	# 4 training hardcoded samples
	training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
	training_set_outputs = array([[0, 1, 1, 0]]).T

	# let's train the network
	neural_network.train(training_set_inputs, training_set_outputs, 2000)

	print("Weights are:\n",neural_network.synaptic_weights)
	print("Outputs is:\n", neural_network.forward(training_set_inputs))

	plt.plot(neural_network.error_history)
	plt.grid()
	plt.show()

if __name__ == '__main__':
	main()