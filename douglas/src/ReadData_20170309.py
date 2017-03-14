
# ========================
# douglas fletcher
# date: 2017.03.09
# purpose: model methods
# ========================

print("reading pandas library...")
from pandas import read_csv, DataFrame


print("importing modelMethod dependencies...")

class ReadData:
	"""
	# douglas fletcher
	# pupose: readdata class:
		Using simpleton pattern 
		create class which is used for 
		reading data, and default methods 
		for getting training & testing data
		returning pandas dataframe.
	# date: 2017.03.09
	"""
	readDataInst = None

	@classmethod
	def __init__(self, fileName, fileLoc):
		# __init singleton
		if not self.readDataInst:
			print("creating object")
			self.readDataInst = self.__ReadData(fileName, fileLoc)
			self.readDataInst.createData()
		else:
			print("getting instance...")

	@classmethod
	def getTraining(self):
		# def get method
		return self.readDataInst.getData()[0]

	@classmethod
	def getTesting(self):
		# def get method
		return self.readDataInst.getData()[1]



	class __ReadData:
		"""
		__init__ 
			as simpleton design pattern	
		"""	

		@classmethod
		def __init__(self, fileName, fileLoc):
			# inputs: filename & location
			# returns: list train & test data
			self.fileName = fileName
			self.fileLoc = fileLoc
			self.outdata = []

		@classmethod
		def createData(self):
			# read data file training & test
			print("reading rawdata")
			try:
				tempRead = read_csv(self.fileLoc + self.fileName)
				rowLen = len(tempRead)
				print("%s rows read from file" %(rowLen))
				print("saving training data - 70% of rawdata")
				self.outdata.append(tempRead[:int(rowLen * 0.7)])
				print("saving test data - 30% of rawdata")			
				self.outdata.append(tempRead[int(rowLen * 0.7):])
			except:
				print("file/location: cannot be found")

		@classmethod
		def getData(self):
			# getdata as list
			return self.outdata
		


print("dependencies ready to use.")

# example: create ReadData object
# read1 = ReadData(RAWDATAFILE, DATALOCATION)
# read1.getTraining()
# read1.getTesting()