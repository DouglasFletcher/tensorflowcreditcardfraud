

# ===========================
# douglas fletcher
# 2017.03.14
# purpose: predict fraudulent
# credit card transactions
# ===========================

# dependencies
import os

# local classes
from src.ReadData_20170309 import ReadData
from src.DataInvestigate_20170309 import DataInvestigate
from src.CreateDataset_20170315 import CreateDataset
from src.CreateSimpleNeuralNet_20170315 import CreateSimpleNeuralNet

# globals
WORKDIR = os.getcwd()
DATALOC = "C:/Users/douglas.fletcher/Documents/projects/2017/riskFraudGroup/creditcardfraud/"
RAWDATA = "creditcard.csv"

# ========================
# train & test data object
# ========================
read1 = ReadData(RAWDATA, DATALOC)
tests = read1.getTesting()
train = read1.getTraining()

# ==================
# data investigation
# ==================
PERIODVAR = "Time"
AMOUNTVAR = "Amount"
LABELSVAR = "Class"
FEATRSVAR = list(set(train.columns).difference(set([PERIODVAR, AMOUNTVAR, LABELSVAR])))
dataInvestObj = DataInvestigate(train, PERIODVAR, AMOUNTVAR, LABELSVAR, FEATRSVAR)
# strange bug after running this method... 
# need to investigate but graphs show fine
#dataInvestObj.distFeaturesByLabel()

# ======================
# create transformations
# ======================
testsTrans = CreateDataset(tests).runTransforms()
trainTrans = CreateDataset(train).runTransforms()

# =======================
# create simple NeuralNet
# =======================
neuralNet = CreateSimpleNeuralNet(trainTrans, testsTrans, 100, 50000).runSimpleNeuralNet()




