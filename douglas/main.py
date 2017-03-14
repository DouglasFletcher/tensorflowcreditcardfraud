

# ===========================
# douglas fletcher
# 2017.03.09
# purpose: predict fraudulent
# credit card transactions
# ===========================

# dependencies
import os
from src.ReadData_20170309 import ReadData

# globals
WORKDIR = os.getcwd()
DATALOC = "C:/Users/douglas.fletcher/Documents/projects/2017/riskFraudGroup/creditcardfraud/"
RAWDATA = "creditcard.csv"

# train & test data object
# reference as:
#	instance.getTraining()
#	instance.getTesting()
read1 = ReadData(RAWDATA, DATALOC)


#print("reading rawdata")
#READRAW = read_csv(DATALOC + RAWDATA)









