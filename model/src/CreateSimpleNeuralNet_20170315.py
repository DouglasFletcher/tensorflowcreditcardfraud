


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
		self.learningRate = 0.01
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
		1. define input/output & layers: 
			these are vectors/matrices which define input/output vector 
			placeholders, and also the hidden layer transformations as 
			placeholders - tensorflow defines the dataflows first before 
			the model is run
		2. define tensor model: 
			tensor flow defines the network logic first before the model is 
			run. In this case, l1 is a 18 node with linear transformation with 
			18 weights & bias (e.g. x * weights + bias) for each node. Next a 
			sigmoid transformation. l2 transformation is linear transformation 
			on l1 output with weights & biases but then softmax transformation 
			(this will give normalized probability values e.g. [0.8, 0.2])
		3. define optimization specifications: 
			optimize weights with feedback loop (backpropagation) using predefined 
			tensor flow methods.
		4. save defined model 
		"""
		print("\tdefining neural net")
		# 1. define input/output & layers
		trainIn = tf.placeholder("float", [None, self.nodes[0]])
		trainOut = tf.placeholder("float")
		hiddenLayer1 = {
			  "weights": tf.Variable(tf.random_normal([self.nodes[0], self.nodes[0]]))
			, "biases": tf.Variable(tf.zeros([self.nodes[0]]))
		}
		outputLayer = {
			"weights": tf.Variable(tf.random_normal([self.nodes[0], 2]))
			, "biases": tf.Variable(tf.zeros([2]))
		}
		# 2. define tensor model
		l1 = tf.add(tf.matmul(trainIn, hiddenLayer1["weights"]), hiddenLayer1["biases"])
		l1 = tf.sigmoid(l1)
		l2 = tf.add(tf.matmul(l1, outputLayer["weights"]), outputLayer["biases"])
		output = tf.nn.softmax(l2)
		# 3. define optimization specifications 
		#error = tf.sub(output, trainOut)
		#cost = tf.reduce_mean(tf.square(error))
		#optimizer = tf.train.GradientDescentOptimizer(self.learningRate).minimize(cost)
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, trainOut))
		optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(cost)
		# 4. save defined model
		self.defModel = [trainIn, trainOut, output, cost, optimizer]
		

	@classmethod
	def __runNeuralNet(self):
		"""
		run defined model
		1. define tensorflow session:
			the session to tensorflow is created for modelling
		2. prepare input data:
			test data is prepared to be fed into the predefined tensor
			model. test features fed in should be list of lists of size
			18 (given there are 18 features). Test labels should be of no 
			particular size, but in this instance I have used 'one-hot' 
			a list of 2 where each element represents probability of label  
		3. get saved tensors:
			model defined in __defineNeuralNet is used
		4. train model:
			train model by epoch & batches. I have defined the sampling batch
			as just taking non-replacement sampling of all the dataset, so
			at least every data point is sampled once. Note the data was 
			randomly ordered in 2.
		5. save results
		"""
		print("\trunning neural network")
		# 1. define tensorflow session
		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)
		# 2. prepare input data
		testsShuffle = self.testset.sample(frac=1)
		trainOutAct = testsShuffle["fraud"].apply(lambda s: [0,1] if s == 0 else [1,0]).to_frame()
		trainAct = concat([testsShuffle[list(testsShuffle.columns[0:18])], trainOutAct], axis=1)
		# 3. get saved tensors
		trainIn, trainOut = self.defModel[0], self.defModel[1]
		output, costAndOpt = self.defModel[2], self.defModel[3:]
		# 4. train model
		for epoch in range(self.trainEpochs):
			epochLoss = 0
			# determine number of batches
			batches = int(len(testsShuffle) / self.batchSize)+1
			for batch in range(batches):
				startPoint, endPoint = batch*self.batchSize, (batch+1)*self.batchSize-1
				#print("\t\t\trunning batch:", batch+1, "of", batches)
				trainBatch = trainAct[startPoint:endPoint]
				trainInBatch  = trainBatch[trainAct.columns[0:18]].values.tolist()
				trainOutBatch = trainBatch[trainAct.columns[18:]].values.tolist()
				# run optimization
				loss, _ = sess.run(
					  costAndOpt
					, feed_dict={trainIn:trainInBatch, trainOut:trainOutBatch}
				)
				epochLoss += loss
			# track progress
			if ((epoch+1) % 10 == 0):
				print("\t\tEpoch", epoch+1, "out of", self.trainEpochs)
				print("\t\tEpoch loss:", epochLoss)
		# 5. save result set
		self.defModel = [trainIn, trainOut, output]+costAndOpt 
		# close session
		sess.close()


	@classmethod
	def __testNeuralNet(self):
		"""
		test defined model
		1. define tensor flow session
		2. prepare test data:
			get test data as test for accuracy
		3. test set predictions:
			use the test data & the trained model to 
			test for the accuracy of the prediction
		4. model accuracy:
			use argmax as way to transform from probabilites 
			in range [0,1] from logistic to binary {0,1} and
			check count of equal allocations 
		5. save predictions:
			output results
		"""
		print("\ttesting neural network")
		# 1. define tensorflow session
		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)
		# 2. prepare test data
		testAct = self.testset
		testFeatures = testAct[testAct.columns[0:18]].values.tolist()
		testLabelAct = testAct["fraud"].apply(
			lambda s: [0,1] if s == 0 else [1,0]
		).tolist()
		# 3. test set predictions
		feed_dict 	  = {self.defModel[0]: testFeatures}
		outTest 	  = sess.run(self.defModel[2], feed_dict).tolist()
		outTestIndex  = sess.run(tf.argmax(outTest,1)).tolist()
		outLabelIndex = sess.run(tf.argmax(testLabelAct,1)).tolist()
		# 4. model accuracy
		accuracy = sess.run(tf.reduce_mean(
			tf.cast(
				tf.equal(outTestIndex, outLabelIndex), "float")
			)
		)
		print("model accuracy:")
		print(accuracy)
		# 5. save predictions
		self.resModel = [outTest, testLabelAct]
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
