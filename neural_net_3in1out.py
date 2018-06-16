from numpy import random, dot, exp, array


class NeuralNetwork():

	def __init__(self):

		random.seed(1)

		self.synaptic_weights = 2 * random.random((3,1)) - 1


	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))


	def __sigmoid_derivative(self, x):
		return x * (1 - x)


	def train(self, given_input, given_output, epochs):

		for i in range(epochs):

			output = self.forward(given_input)

			error = given_output - output

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
	neural_network.train(training_set_inputs, training_set_outputs, 10000)

	print("Weights are: ",neural_network.synaptic_weights)
	print("outputs is: ", neural_network.forward(training_set_inputs))

if __name__ == '__main__':
	main()