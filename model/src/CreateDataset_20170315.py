

# ========================
# douglas fletcher
# date: 2017.03.15
# ========================

print("reading CreateDataset...")
from pandas import Series, concat

class CreateDataset:
	"""
	douglas fletcher
	pupose: create test & training datasets variables. For now 
	use basic neural network. This requires creating binary 
	classification variables based on given data & investition 
	(done in DataInvestigate class for now)	outcomes.
	"""
	@classmethod
	def __init__(self, datasetIn):
		self.datasetIn = datasetIn
		self.datasetOt = None 


	@classmethod
	def __createTransformations(self):
		"""
		create transformations: based on visualizations of fraud transactions 
		distributions (i.e. DataInvestigate.distFeaturesByLabel() method) some 
		variables are either more left or right skewed (relative to non-fraud 
		transactions). This can be used as justification to find instances which 
		differentiate fraud vs non-fraud transactions and create binary classification 
		variables. These can be fed to the first perceptons layer of the neural network 
		as a starting point.
		"""
		# create transformations: all features
		v01trans =  Series(self.datasetIn["V1"].apply(lambda s: 1 if s < -3.4 else 0), name="v01trans")
		v02trans =  Series(self.datasetIn["V2"].apply(lambda s: 1 if s >  2.5 else 0), name="v02trans")
		v03trans =  Series(self.datasetIn["V3"].apply(lambda s: 1 if s < -3.4 else 0), name="v03trans")
		v04trans =  Series(self.datasetIn["V4"].apply(lambda s: 1 if s >  3.8 else 0), name="v04trans")
		v05trans =  Series(self.datasetIn["V5"].apply(lambda s: 1 if s < -3.2 else 0), name="v05trans")
		v06trans =  Series(self.datasetIn["V6"].apply(lambda s: 1 if s < -2.1 else 0), name="v06trans")
		v07trans =  Series(self.datasetIn["V7"].apply(lambda s: 1 if s < -3.5 else 0), name="v07trans")
		v09trans =  Series(self.datasetIn["V9"].apply(lambda s: 1 if s < -2.0 else 0), name="v09trans")
		v10trans = Series(self.datasetIn["V10"].apply(lambda s: 1 if s < -2.9 else 0), name="v10trans")
		v11trans = Series(self.datasetIn["V11"].apply(lambda s: 1 if s >  2.0 else 0), name="v11trans")
		v12trans = Series(self.datasetIn["V12"].apply(lambda s: 1 if s < -2.9 else 0), name="v12trans")
		v14trans = Series(self.datasetIn["V14"].apply(lambda s: 1 if s < -1.9 else 0), name="v14trans")
		v16trans = Series(self.datasetIn["V16"].apply(lambda s: 1 if s < -2.2 else 0), name="v16trans")
		v17trans = Series(self.datasetIn["V17"].apply(lambda s: 1 if s < -2.9 else 0), name="v17trans")
		v18trans = Series(self.datasetIn["V18"].apply(lambda s: 1 if s < -1.3 else 0), name="v18trans")
		v19trans = Series(self.datasetIn["V19"].apply(lambda s: 1 if s >  2.0 else 0), name="v19trans")
		v21trans = Series(self.datasetIn["V21"].apply(lambda s: 1 if s >  1.7 else 0), name="v21trans")
		v27trans = Series(self.datasetIn["V27"].apply(lambda s: 1 if s >  1.3 else 0), name="v27trans")

		# create labels
		fraud = Series(self.datasetIn["Class"], name="fraud")

		# create dataset
		allTrans = concat(
			[
				  v01trans, v02trans, v03trans, v04trans, v05trans, v06trans
				, v07trans, v09trans, v10trans, v11trans, v12trans, v14trans
				, v16trans, v17trans, v18trans, v19trans, v21trans, v27trans
				, fraud
			]
			, axis=1
		).reset_index()
		# drop index
		del allTrans["index"]
		# save
		self.datasetOt = allTrans


	@classmethod 
	def __getDataTrans(self):
		"""
		get method datatrans
		"""
		return self.datasetOt


	@classmethod
	def runTransforms(self):
		"""
		run all process
		"""
		print("creating transform dataset...")
		self.__createTransformations()
		return self.__getDataTrans()


