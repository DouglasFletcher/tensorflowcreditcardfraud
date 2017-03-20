


# ========================
# douglas fletcher
# date: 2017.03.15
# ========================

print("reading CreateSimpleNeuralNet...")

import tensorflow as tf
from pandas import concat

class CreateSimpleNeuralNet:
	"""
	douglas fletcher
	purpose: create simple neural net 2 layer perceptron layer
	with variable inputs using tensorflow package. 
	"""

	@classmethod
	def __init__(self, trainset, testset, trainEpochs, batchSize):
		"""
		initialize variables
		"""
		# input data
		self.trainset = trainset
		self.testset = testset
		# model parameters
		self.trainEpochs = trainEpochs
		self.batchSize = batchSize
		self.learningRate = 0.001
		# network structure: need to make dynamic 
		# (__defineNeuralNet has dependency)
		self.nodes = [18]
		# result set
		self.defModel = None
		self.resModel = None


	'''
	@classmethod
	def __defineNeuralNet(self):
		"""
		define neural net with tensorflow
		"""
		print("\tdefining neural net")
		# define input/output definitions
		xVals = tf.placeholder("float", [None, 18])
		yVals = tf.placeholder("float")
		# define tensorflow layers with default values random
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
		self.defModel = [xVals, yVals, prediction, cost, optimizer]
	'''



	@classmethod
	def __defineNeuralNet(self):
		"""
		define neural net with tensorflow
		"""
		def step(x):
			is_greater = tf.greater(x, 0)
			as_float = tf.to_float(is_greater)
			doubled = tf.mul(as_float, 2)
			return tf.sub(doubled, 1)

		print("\tdefining neural net")
		# define weights
		weights   = tf.Variable(tf.random_normal([18, 1]))
		train_in  = self.testset[list(self.testset.columns[0:18])].values 
		train_out = self.testset[list(self.testset.columns[18:])].values
		# define model output
		output = step(tf.matmul(train_in, weights))
		errors = tf.sub(train_out, output)
		mse = tf.reduce_mean(tf.square(errors))
		# update parameters
		delta = tf.matmul(train_in, error, transpose_a=True)
		train = tf.assign(w, tf.add(w, delta))
		# create session
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		# optimization
		err, target = 1, 0
		epoch, max_epoch = 0, 10
		while err > target and epoch < max_epoch:
			epoch += 1
			err, _ = sess.run([mse, train])
			print("epoch:", epoch, "mse:", err)



	@classmethod
	def __runNeuralNet(self):
		"""
		run defined model
		"""
		print("\trunning neural network")
		# define tensorflow session
		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)
		# shuffle training dataset
		testsShuffle = self.testset.sample(frac=1)
		# features & labels (as "one-hot")
		xIn = self.testset[list(self.testset.columns[0:18])]
		yIn = self.testset["fraud"].apply(
			lambda s: [0,1] if s == 0 else [1,0]
		).to_frame()
		# both
		xyIn = concat([xIn, yIn], axis=1)
		# get saved model
		xVals, yVals = self.defModel[0], self.defModel[1]
		modFrame = self.defModel[3:]
		# train model: use defined epochs to train
		for epoch in range(self.trainEpochs):
			print("\t\tEpoch", epoch+1, "out of", self.trainEpochs)
			epochLoss = 0
			# run in batches
			batches = int(len(testsShuffle) / self.batchSize)+1
			for batch in range(batches):
				startPoint, endPoint = batch*self.batchSize, (batch+1)*self.batchSize-1
				# batch data
				print("\t\t\trunning batch:", batch+1, "of", batches)
				# get sample: i will just take in order
				xySample = xyIn[startPoint:endPoint]
				xepoch = xySample[self.testset.columns[0:18]].values
				yepoch = xySample[self.testset.columns[18:]].values
				# run optimization
				#print(xepoch[0:3])
				#print(yepoch[0:3])
				val, loss = sess.run(modFrame, feed_dict={xVals: xepoch, yVals: yepoch})
				epochLoss += loss
			print("\t\tEpoch loss:", epochLoss)
		'''
		'''
		# close session
		sess.close()


	@classmethod
	def __testNeuralNet(self):
		"""
		test defined model
		"""
		print("\ttesting neural network")


	@classmethod
	def __getOutput(self):
		"""
		return model output
		"""
		return self.resModel


	@classmethod
	def runSimpleNeuralNet(self):
		"""
		run class methods
		"""
		print("\ncreating simple neuralnet")
		self.__defineNeuralNet()
		#self.__runNeuralNet()
		#self.__testNeuralNet()
		result = self.__getOutput()
		return result
