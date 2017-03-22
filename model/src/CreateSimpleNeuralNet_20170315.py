


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
	purpose: 
		create simple neural net 2 layer 
		perceptron layer using tensorflow package 
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


	@classmethod
	def __defineNeuralNet(self):
		"""
		define neural net with tensorflow
		1. define input/output tensors: 
			these are vectors/matrices which define input/output vector 
			placeholders
		2. define mdoel layers: 
			define the hidden layer transformations as	placeholders - 
			tensorflow defines the dataflows first before the model is run
		3. define flow model: 
			tensor flow defines the network logic first before the model is 
			run. In this case, l1 is a 18 node with linear transformation with 
			18 weights & bias (e.g. x * weights + bias) for each node. Next a 
			sigmoid transformation. l2 transformation is linear transformation 
			on l1 output with weights & biases but then softmax transformation 
			(this will give normalized probability values e.g. [0.8, 0.2])
		4. define optimization specifications: 
			define optimization cost function & optimization method - prebuilt 
			methods from tensorflow
		5. save defined model:
			defModel, now defined, can be referenced 
		"""
		print("\tdefining neural net")
		# 1. define input/output tensors
		tensorIn = tf.placeholder("float", [None, self.nodes[0]])
		tensorOut = tf.placeholder("float")
		# 2. define model layers
		hiddenLayer1 = {
			  "weights": tf.Variable(tf.random_normal([self.nodes[0], self.nodes[0]]))
			, "biases": tf.Variable(tf.zeros([self.nodes[0]]))
		}
		outputLayer = {
			"weights": tf.Variable(tf.random_normal([self.nodes[0], 2]))
			, "biases": tf.Variable(tf.zeros([2]))
		}
		# 3. define flow of model
		l1 = tf.add(tf.matmul(tensorIn, hiddenLayer1["weights"]), hiddenLayer1["biases"])
		l1 = tf.sigmoid(l1)
		l2 = tf.add(tf.matmul(l1, outputLayer["weights"]), outputLayer["biases"])
		output = tf.nn.softmax(l2)
		# 4. define optimization specifications 
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, tensorOut))
		optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(cost)
		# 5. save defined model
		self.defModel = [tensorIn, tensorOut, output, cost, optimizer]
		

	@classmethod
	def __runNeuralNet(self):
		"""
		run defined model
		1. define tensorflow session:
			tensorflow session created - this is a reference when 
			running any data through the defined model
		2. prepare input data:
			create a handel to features and labels as reference  
		3. get saved tensors:
			tensors defined in __defineNeuralNet
		4. train model:
			train model by epoch & batches. I have defined the sampling batch
			as just taking non-replacement sampling of all the dataset, so
			at least every data point is sampled once. Note the data was 
			randomly ordered dataprocessing.
		5. save results:
			train model parameters saved to be used for test data 
			to check accuracy
		"""
		print("\trunning neural network")
		# 1. define tensorflow session
		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)
		# 2. prepare input data
		features = self.trainset[self.trainset.columns[0:18]].values.tolist()
		labels   = self.trainset.ix[:,18].values.tolist()
		# 3. get saved tensors
		defModel = self.defModel
		# 4. train model
		for epoch in range(self.trainEpochs):
			epochLoss = 0
			# determine number of batches
			batches = int(len(self.trainset) / self.batchSize)+1
			for batch in range(batches):
				# define batch sample range
				batchSample = [batch*self.batchSize, (batch+1)*self.batchSize-1]
				# batch labels & features
				labelsBatch  = labels[batchSample[0]:batchSample[1]] 
				featureBatch = features[batchSample[0]:batchSample[1]] 
				# run optimization
				loss, _ = sess.run(
					  defModel[3:]
					, feed_dict={defModel[0]:featureBatch, defModel[1]:labelsBatch}
				)
				epochLoss += loss
			# track loss & accuracy
			if ((epoch+1) % 10 == 0):
				# loss function
				print("\t\tEpoch", epoch+1, "out of", self.trainEpochs)
				print("\t\t\tEpoch loss:", epochLoss)
				# epoch accuracy
				feedDictEpoch = {defModel[0]: features}
				epochPrediction = sess.run(defModel[2], feedDictEpoch).tolist()
				accuracy = sess.run(tf.reduce_mean(
						tf.cast(tf.equal(tf.argmax(epochPrediction,1), tf.argmax(labels,1)
						),"float")))
				print("\t\t\tEpoch Model accuracy:",round(accuracy*100,2),"%")
		# 5. save result set
		self.defModel = defModel 
		sess.close()



	@classmethod
	def __testNeuralNet(self):
		"""
		test defined model
		1. define tensor flow session
		2. prepare test data:
			get test data labels & features
		3. prediction on testset:
			use the test data & the trained model to test for the 
			accuracy of the prediction
		4. model accuracy:
			use argmax get index of most likely prediction e.g. [0.2,0.8] => index 
			1 of actual vs predicted to check for model accuracy - same calculation 
			as epoch prediction 
		5. save predictions:
			output results
		"""
		print("\ttesting neural network")
		# 1. define tensorflow session
		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)
		# 2. prepare test data
		testFeatures = self.testset[self.testset.columns[0:18]].values.tolist()
		testLabels   = self.testset.ix[:,18].values.tolist()
		# 3. prediction on testset
		feed_dict = {self.defModel[0]: testFeatures}
		testPrediction = sess.run(self.defModel[2], feed_dict).tolist()
		# 4. model accuracy
		accuracy = sess.run(tf.reduce_mean(
			tf.cast(tf.equal(tf.argmax(testPrediction,1), tf.argmax(testLabels,1)), "float")
		))
		print("\t\tModel accuracy:", round(accuracy *100,2), "%")
		# 5. save predictions
		self.resModel = [testPrediction, testLabels]
		sess.close()


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
		self.__runNeuralNet()
		self.__testNeuralNet()
		result = self.__getOutput()
		return result
