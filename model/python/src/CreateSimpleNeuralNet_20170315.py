


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
			define optimization cost function & optimization method - tried different 
			cost functions e.g. softmax/squared and/or l2 regularization (adding a penalty 
			on the norm of the weights to the loss)
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
		regularizers = tf.nn.l2_loss(hiddenLayer1["weights"]) + tf.nn.l2_loss(outputLayer["weights"])
		costwithReg = tf.reduce_mean(cost + 0.01 * regularizers)
		#costwithReg = cost
		optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(costwithReg)
		# 5. save defined model
		self.defModel = [tensorIn, tensorOut, output, costwithReg, optimizer]


	@staticmethod
	def __calcPredAccuracy(inSess, inDefModel, inFeatures, inLabels, setIn):
		"""
		static method to calculate prediction accuracy
		1. predict labels:
			using current model weights model model prediction
		2. predict accuracy:
			calculate prediction on accuracy
		"""
		# 1. predict label
		feedDict = {inDefModel[0]: inFeatures}
		pred = inSess.run(inDefModel[2], feedDict).tolist()
		# 2. predict accuracy
		accuracy = inSess.run(tf.reduce_mean(
			tf.cast(tf.equal(tf.argmax(pred,1), tf.argmax(inLabels,1)
		), "float")))
		print("\t\t\t",setIn,"accuracy:",round(accuracy*100,2),"%")
		# 3. return predictions
		return [feedDict, pred, accuracy]


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
		4. train model & report progress:
			4.1 train model: 
				train model by epoch & batches. I have defined the sampling batch
				as just taking non-replacement sampling of all the dataset, so
				at least every data point is sampled once. Note the data was 
				randomly ordered in dataprocessing.
			4.2 report progress of model:
				4.2.1 report epoch loss:
					print epoch loss
				4.2.2 report training accuracy:
					print training accuracy
				4.2.3 report testing accuracy:
					print training accuracy on test set by 
					taking a random sample
		5. save results:
			train model parameters saved to be used for test data 
			to check accuracy
		"""
		print("\trunning neural network")
		# 1. define tensorflow session
		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)
		# 2. prepare test & train input data
		features = self.trainset[self.trainset.columns[0:18]].values.tolist()
		labels   = self.trainset.ix[:,18].values.tolist()
		# 3. get saved tensors
		defModel = self.defModel
		# 4. train & report progress of model
		for epoch in range(self.trainEpochs):
			# 4.1 train model
			epochLoss = 0
			batches = int(len(self.trainset) / self.batchSize)+1
			for batch in range(batches):
				batchSample = [batch*self.batchSize, (batch+1)*self.batchSize-1]
				labelsBatch  = labels[batchSample[0]:batchSample[1]] 
				featureBatch = features[batchSample[0]:batchSample[1]] 
				loss, _ = sess.run(
					  defModel[3:]
					, feed_dict={defModel[0]:featureBatch, defModel[1]:labelsBatch}
				)
				epochLoss += loss
			# 4.2 report progress of model
			if ((epoch+1) % 10 == 0) or (epoch == self.trainEpochs):
				# 4.2.1 report epoch loss
				print("\t\tEpoch", epoch+1, "out of", self.trainEpochs)
				print("\t\t\tEpoch loss:", epochLoss)
				# 4.2.2 report training accuracy
				outTrain = self.__calcPredAccuracy(sess, defModel, features, labels, "Epoch training")
				# 4.2.3 report testing accuracy
				testSample = self.testset.sample(frac=0.1)
				testFeatures = testSample[testSample.columns[0:18]].values.tolist()
				testLabels   = testSample.ix[:,18].values.tolist()
				outTest = self.__calcPredAccuracy(sess, defModel, testFeatures, testLabels, "Epoch testing")
		# 5. report testing accuracy
		print("\ttesting accuracy of neural network")
		testTotSample 	= self.testset.sample(frac=1)
		testTotFeatures = testTotSample[testTotSample.columns[0:18]].values.tolist()
		testTotLabels   = testTotSample.ix[:,18].values.tolist()
		testTotOut = self.__calcPredAccuracy(sess, defModel, testTotFeatures, testTotLabels, "Total testing")
		# 6. save model & results
		self.defModel = defModel
		self.resModel = [testTotOut[0], testTotLabels]		
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
		result = self.__getOutput()
		return result
