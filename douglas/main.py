

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
#dataInvestObj.distFeaturesByLabel()

# different distributions
#[,"V2","V3","V4","V5","V6","V7","V9","V10","V11","V12","V14","V16","V17","V18","V19"]

