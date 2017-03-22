
# ========================
# douglas fletcher
# date: 2017.03.09
# ========================

print("reading ReadData...")

from pandas import read_csv

class ReadData:
	"""
	# douglas fletcher
	# pupose: readdata class:
		Using simpleton pattern create class which is used for 
		reading data, and default methods for getting training 
		& testing data returning pandas dataframe.
	# date: 2017.03.09
	"""
	readDataInst = None

	@classmethod
	def __init__(self, fileName, fileLoc):
		# __init singleton
		if not self.readDataInst:
			self.readDataInst = self.__ReadData(fileName, fileLoc)
			self.readDataInst.__createData()

	@classmethod
	def getTraining(self):
		# def get method
		print("\treturning raw training data")
		return self.readDataInst.getData()[0]

	@classmethod
	def getTesting(self):
		# def get method
		print("\treturning raw testing data")
		return self.readDataInst.getData()[1]

	class __ReadData:
		"""
		__init__ : as simpleton design pattern	
		"""	

		@classmethod
		def __init__(self, fileName, fileLoc):
			# inputs: filename & location
			# returns: list train & test data
			self.fileName = fileName
			self.fileLoc = fileLoc
			self.outdata = []

		@classmethod
		def __createData(self):
			# read data file training & test
			print("\nreading rawdata")
			try:
				tempRead = read_csv(self.fileLoc + self.fileName)
				rowLen = len(tempRead)
				print("\t%s rows read from file" %(rowLen))
				# train set
				print("\tsaving training data - 70% of rawdata")
				train = tempRead[:int(rowLen * 0.7)]
				train.name = "train"
				self.outdata.append(train)
				# test
				print("\tsaving test data - 30% of rawdata")			
				tests = tempRead[int(rowLen * 0.7):]
				tests.name = "tests"
				self.outdata.append(tests)
			except:
				print("\tfile/location: cannot be found")

		@classmethod
		def getData(self):
			# getdata as list
			return self.outdata
		

# example: create ReadData object
# read1 = ReadData(RAWDATAFILE, DATALOCATION)
# read1.getTraining()
# read1.getTesting()