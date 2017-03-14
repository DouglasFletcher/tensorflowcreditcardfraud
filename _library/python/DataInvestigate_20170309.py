
# ========================
# douglas fletcher
# date: 2017.03.15
# ========================

import seaborn as sns
import matplotlib.pyplot as plt


class DataInvestigate:
	"""
	douglas fletcher
	pupose: create data investigation
	i.e. distribution of features & 
	amounts by each label add methods  
	as questions arise :)
	"""
	@classmethod
	def __init__(self, dataset, periodvar, amountvar, labelsvar, featrsvar):
		"""
		dataset has datatype categories:
		reference for later use
		"""
		self.dataset = dataset
		self.periodvar = periodvar
		self.amountvar = amountvar 
		self.labelsvar = labelsvar
		self.featrsvar = featrsvar


	@classmethod
	def distFeaturesByLabel(self):
		"""
		look at distributions of 
		feature variables vs the
		labels variable to see 
		differences in the data
		"""
		print("\tploting distribution of features by label")
		# setup plotting canvas
		plt.figure()
		# create datasubsets
		for index, feature in enumerate(self.dataset[self.featrsvar]):
			# create dataset
			class0=self.dataset[feature][self.dataset[self.labelsvar]==0].tolist()
			class1=self.dataset[feature][self.dataset[self.labelsvar]==1].tolist()
			# create plots
			plt.subplots(ncols=1)
			sns.distplot(class0, bins = 60, hist=False)
			sns.distplot(class1, bins = 60, hist=False)	
			# axis labels
			plt.xlabel("feature value")
			plt.suptitle("distribution of feature: %s" %(feature))			
			# show
			plt.show()



# example: create DataInvestigate object
#PERIODVAR = "Time"
#AMOUNTVAR = "Amount"
#LABELSVAR = "Class"
#FEATRSVAR = ["V1","V2"]
#dataInvestObj = DataInvestigate(train, PERIODVAR, AMOUNTVAR, LABELSVAR, FEATRSVAR)
#dataInvestObj.distFeaturesByLabel()







