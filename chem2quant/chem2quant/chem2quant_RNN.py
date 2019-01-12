import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import RDConfig

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

import psycopg2 # postgres module

"""

Creating the neural network.

The placeholder X will act as an input layer. During the execution phase, it will be replaced with 
one training one training batch at a time (note that all the instances in a training batch 
will be processed simultaneously by the neural network).
 
"""


try:
    conn = psycopg2.connect("dbname='emolecules' user='sang' host='localhost' password='silver!!'")
except:
    print("Did not connect to database")

cur = conn.cursor()
cur.execute("SELECT * FROM raw_data LIMIT 1000;")
cur.execute("SELECT DISTINCT smiles, emol_id from raw_data LIMIT 1000;")
database = cur.fetchall()  
    
def vectorize(smiles):
        """
        Changing the smiles representation into a one-hot representation
        """
        one_hot =  np.zeros((smiles.shape[0], embed , len(charset)), dtype=np.int8)
        for i,smile in enumerate(smiles):
            #encode the startchar
            one_hot[i,0,char_to_int["!"]] = 1
            #encode the rest of the chars
            for j,c in enumerate(smile):
                one_hot[i,j+1,char_to_int[c]] = 1
            #Encode endchar
            one_hot[i,len(smile)+1:,char_to_int["E"]] = 1
        #Return two, one for input and the 2other for output
        return one_hot[:,0:-1,:], one_hot[:,1:,:]

file = "/home/noh/Desktop/QM_MM/chem2quant/chem2quant/smifiles/gdb11_size08.smi"
data = pd.read_csv(file, delimiter = "\t", names = ["smiles","No","Int"])

smiles_train, smiles_test = train_test_split(data["smiles"], random_state=42)

charset = set("".join(list(data.smiles))+"!E")
char_to_int = dict((c,i) for i,c in enumerate(charset))
int_to_char = dict((i,c) for i,c in enumerate(charset))
embed = max([len(smile) for smile in data.smiles]) + 5

X_train, Y_train = vectorize(smiles_train.values)
X_test,Y_test = vectorize(smiles_test.values)

smiles_len = len(X_train[0])

# ---------------------------#
# Neural network below here  #
# ---------------------------#

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

## TODO - to finish this part - How do we timewise-slice the one-hot representations of the molecules?

#n_epochs = 100
#batch_size = 150

#with tf.Session() as sess:
#    init.run()
#    for epoch in range(n_epochs):
#        for iteration in range(mnist.train.num_examples // batch_size):
#            X_batch, y_batch = mnist.train.next_batch(batch_size)
#            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
#            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
#            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
#    print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
