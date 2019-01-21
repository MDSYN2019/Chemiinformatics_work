"""
Tensorflow implementation of 

"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

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

# Getting smiles from pubchempy

import pubchempy as pcp
from pubchempy import Compound, get_compounds

n_hidden1 = 300
n_hidden2 = 300
n_hidden3 = 300

learning_rate = 0.01
n_outputs = 3 # We have 3 output classes
datadir = '/home/noh/Desktop/Current_work_in_progress/Chemiinformatics/RDKIT/rdkit/Docs/Book/data'

"""
Creating the neural network

The placeholder X will act as an input layer. During the execution phase, it will be replaced with 
one training one training batch at a time (note that all the instances in a training batch 
will be processed simultaneously by the neural network).
 
"""

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
    np_fps.append(arr)
    return np_fps
def mol2arr(mol):
  arr = np.zeros((1,))
  fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
  DataStructs.ConvertToNumpyArray(fp, arr)
  return arr

train_mol = [mol for mol in Chem.SDMolSupplier(os.path.join(datadir,'solubility.train.sdf')) if mol != None]
test_mol = [mol for mol in Chem.SDMolSupplier(os.path.join(datadir,'solubility.test.sdf')) if mol != None]
cls_mol = list(set([mol.GetProp('SOL_classification') for mol in train_mol]))
cls_dic = {}

for i, cl in enumerate(cls_mol):
    cls_dic[cl] = i
tf.reset_default_graph()
# make train X, y and test X, y
train_X = np.array([mol2arr(mol) for mol in train_mol])
train_y = np.array([cls_dic[mol.GetProp('SOL_classification')] for mol in train_mol])

test_X = np.array([mol2arr(mol) for mol in test_mol])
test_y = np.array([cls_dic[mol.GetProp('SOL_classification')] for mol in test_mol])

# Set up parameters for the neural network

X = tf.placeholder(tf.float32, shape = (None, np.shape(train_X[0])[0]), name = "X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
he_init = tf.contrib.layers.variance_scaling_initializer()

with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, weights_initializer = he_init, scope = "hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, weights_initializer = he_init, scope = "hidden2")
    logits = fully_connected(hidden2, n_outputs, weights_initializer = he_init, scope = "outputs", activation_fn = None)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
    loss = tf.reduce_mean(xentropy, name = "loss")
    
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y ,1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()
n_epochs = 20
batch_size = 100

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(len(train_X // batch_size)):
            X_batch, y_batch = next_batch(batch_size, train_X, train_y)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: test_X, y: test_y})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    save_path = saver.save(sess, "./my_model_final.ckpt")




