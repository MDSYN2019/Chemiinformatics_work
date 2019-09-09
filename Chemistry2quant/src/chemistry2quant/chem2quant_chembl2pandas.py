


"""
Python module which returns the chembl database as a pandas table, depending on the 
specifications you would like in the end. 

Make sure that the permissions settings on the table has been granted to the USER.
"""

#try:
#	from __future__ import division
#	from __future__ import print_function
#	from __future__ import unicode_literals
#except importError:
#	print("Placeholder")

import sys
import os 
import numpy as np
import pandas as pd 
#import tensorflow as tf

#import deepchem as dc
#from deepchem.models.tensorgraph.models.graph_models import GraphConvModel


try:
    import psycopg2
except:
    print("Error: You need psycopg2 to run this code")
    
class chemblConnect:
    def __init__(self, database, user, host, password):
        self.database = str(database)
        self.user = str(user)
        self.host = str(host)
        self.password = str(password) 
        self.string = "dbname={} user={} host={} password={}".format(self.database, self.user, self.host, self.password)
        self.conn = psycopg2.connect(self.string)
    def issue_command(self):
        pass

# Working with the emolecules module.
    
try:                                                                                                                                                 
    conn = psycopg2.connect(dbname='emolecules', user='sang', host='localhost', password='silver!!', port=5432) 
except:                                                                                                                                              
    print("Did not connect to database")                                                                                                             

cur = conn.cursor()                                                                                                                                  
cur.execute("SELECT * FROM raw_data LIMIT 100;")                                                                                                    
cur.execute("SELECT DISTINCT smiles, emol_id from raw_data LIMIT 100;")                                                                             
database = cur.fetchall()            


"""
Step 1 -> Understand and modify the graph convolution for Tox21 
"""
