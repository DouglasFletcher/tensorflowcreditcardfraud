
import tensorflow as tf


class CreateSimpleNeuralNet:
	"""
	douglas fletcher
	purpose: create simple neural net 2 layer perceptron layer
	with variable inputs using tensorflow package. 
	"""
	@classmethod
	def __init__(self, datasetIn):
		# dataset
		self.datasetIn = datasetIn
		self.output = 1
		# default Parameters
		self.learningRate = 0.001
		self.trainingEpochs = 15
		self.batchSize = 100
		self.display_step = 1
		# default network parameters
		self.hidden1 = 18
		self.hidden2 = 18
		self.nclasses = 2


	@classmethod
	def __createNeuralNet(self):
		"""
		neural net using tensorflow
		"""
		pass


	@classmethod
	def __getOutput(self):
		"""
		return model output
		"""
		return self.output


	@classmethod
	def runSimpleNeuralNet(self):
		"""
		run class methods
		"""
		print("creating simple neural net model...")
		self.__createNeuralNet()
		return self.__getOutput()
