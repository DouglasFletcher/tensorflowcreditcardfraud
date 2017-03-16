


# ========================
# douglas fletcher
# date: 2017.03.15
# ========================

print("reading CreateSimpleNeuralNet...")

import tensorflow as tf

class CreateSimpleNeuralNet:
	"""
	douglas fletcher
	purpose: create simple neural net 2 layer perceptron layer
	with variable inputs using tensorflow package. 
	"""
	@classmethod
	def __init__(self, datasetIn):
		"""
		initialize variables
		"""
		self.datasetIn = datasetIn
		self.defModel = None
		# learning parameters default
		self.learningRate = 0.001
		self.trainEpochs = 5
		self.batchSize = 100
		self.display_step = 1
		# network structure: need to make dynamic 
		# (__defineNeuralNet has dependency)
		self.nodes = [18]


	@classmethod
	def __defineNeuralNet(self):
		"""
		define neural net with tensorflow
		"""
		# define input/output definitions
		xVals = tf.placeholder("float", [None, 18])
		yVals = tf.placeholder("float")
		# define tensorflow layers
		# [1,0,1,....,1] list of len 18 binaries {0, 1}
		hiddenLayer1 = {
			  "weights": tf.Variable(tf.random_normal([18, self.nodes[0]]))
			, "biases": tf.Variable(tf.random_normal([self.nodes[0]]))
		}
		# output layer: {0, 1}
		outputLayer = {
			"weights": tf.Variable(tf.random_normal([self.nodes[0], 1]))
			, "biases": tf.Variable(tf.random_normal([1]))
		}		
		# define model: data * weights + biases
		l1 = tf.add(tf.matmul(xVals, hiddenLayer1["weights"]), hiddenLayer1["biases"])
		l1 = tf.sigmoid(l1)
		# define result
		prediction = tf.add(tf.matmul(l1, outputLayer["weights"]), outputLayer["biases"])
		# define optimization technique
		cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, yVals))
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(cost)
		# save model
		self.defModel = [prediction, cost, optimizer]


	@classmethod
	def __runNeuralNet(self):
		"""
		run defined model
		"""
		# define tensorflow session
		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)
		# training dataset
		#y = self.datasetIn[""]
		# train model
		for epoch in range(self.trainEpochs):
			epochLoss = 0
			#for batch in range(int(SOMETHING / self.batchSize)):
			#inX, inY = traindata.next_batch(self.batchSize)
			epochLoss += 1
			print("Epoch", epoch, "completed out of", self.trainEpochs, "loss:", epochLoss)

		# close session
		sess.close()


	@classmethod
	def __getOutput(self):
		"""
		return model output
		"""
		return self.defModel


	@classmethod
	def runSimpleNeuralNet(self):
		"""
		run class methods
		"""
		print("creating simple neural net model...")
		self.__defineNeuralNet()
		self.__runNeuralNet()
		result = self.__getOutput()
		return result
