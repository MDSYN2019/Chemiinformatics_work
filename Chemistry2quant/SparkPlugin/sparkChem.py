import pandas as pd
import tensorflow as tf
from scipy import io
from pyspark import SparkConf, SparkContext
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import *

# load data

"""

Data taken from the tox21 website 

"""
y_tr = pd.read_csv('tox21_labels_train.csv.gz', index_col=0, compression="gzip")
y_te = pd.read_csv('tox21_labels_test.csv.gz', index_col=0, compression="gzip")
x_tr_dense = pd.read_csv('tox21_dense_train.csv.gz', index_col=0, compression="gzip").values
x_te_dense = pd.read_csv('tox21_dense_test.csv.gz', index_col=0, compression="gzip").values
x_tr_sparse = io.mmread('tox21_sparse_train.mtx.gz').tocsc()
x_te_sparse = io.mmread('tox21_sparse_test.mtx.gz').tocsc()

"""
Load smiles data onto Spark - 
"""

# Make sure to refresh to have a fresh spark session
#try:
#    sc.stop()
#except ValueError:
#    print ("We are not running a sc session at the moment")
#    pass

# ------------------------#
# Boilerplate Spark Stuff #
# ------------------------#

conf = SparkConf().setMaster("local").setAppName("SparkSmiles")
sc = SparkContext(conf = conf)
sqlCtx = SQLContext(sc)

colNames = [str(i) for i in range(0,801,1)]
x_tr_dense_pd = pd.DataFrame(x_tr_dense)
x_tr_spark = sqlCtx.createDataFrame(x_tr_dense_pd, schema = colNames) # Example of a pyspark dataframe created from 
y_tr_spark = sqlCtx.createDataFrame(y_tr) # Example of a pyspark dataframe created from 

# ------------------------------------------------------------ #

#Tox21RawData = sc.textFile("tox21_compoundData.csv")
#header = Tox21RawData.first()
#Tox21RawData.filter(lambda x:x != header)
#csvData = Tox21RawData.map(lambda x: x.split(","))




