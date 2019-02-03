import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

from functools import partial

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

# to make this notebook's output stable across runs

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
from sklearn.model_selection import train_test_split

"""

Creating the neural network.

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

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
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

file = "data/gdb11_size11.smi"
data = pd.read_csv(file, delimiter = "\t", names = ["smiles","No","Int"])
smiles_train, smiles_test = train_test_split(data["smiles"], random_state=42)

n_epochs = 100
batch_size = 150

charset = set("".join(list(data.smiles))+"!E")
char_to_int = dict((c,i) for i,c in enumerate(charset))
int_to_char = dict((i,c) for i,c in enumerate(charset))
embed = max([len(smile) for smile in data.smiles]) + 5

X_train, Y_train = vectorize(smiles_train.values)
X_test, Y_test = vectorize(smiles_test.values)
X_train, Y_train = vectorize(smiles_train.values)

X_train = [a.flatten() for a in X_train]
X_test = [a.flatten() for a in X_test]
Y_train = [a.flatten() for a in Y_train]
Y_test = [a.flatten() for a in Y_test]

#class_1, class2 = np.unique(Y_train, axis = 0)
#smiles_len = len(X_train[0])

# -------------------------------#
#  Neural network - autoencoder  #
# -------------------------------#

n_inputs = 27 * 22
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.001
l2_reg = 0.0001

n_hidden1 = 300
n_hidden2 = 150 # codings

X = tf.placeholder(tf.float32, [None, n_inputs])
he_init = tf.contrib.layers.variance_scaling_initializer() # He initialization
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)

my_dense_layer = partial(tf.layers.dense,
                         activation=tf.nn.elu,
                         kernel_initializer=he_init,
                         kernel_regularizer=l2_regularizer)

hidden1 = my_dense_layer(X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2)
hidden3 = my_dense_layer(hidden2, n_hidden3)
outputs = my_dense_layer(hidden3, n_outputs, activation=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_epochs = 5
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = len(X_train) // batch_size
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="") # not shown in the book
            sys.stdout.flush()                                          # not shown
            X_batch, y_batch = next_batch(batch_size, X_train, Y_train)
            sess.run(training_op, feed_dict={X: X_batch})
        loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})   # not shown
        print("\r{}".format(epoch), "Train MSE:", loss_train)           # not shown
        #saver.save(sess, "./my_model_all_layers.ckpt")                  # not shown


#y = tf.placeholder(tf.int32, [None])

#basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
#outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
#logits = tf.layers.dense(states, n_outputs)
#xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

#loss = tf.reduce_mean(xentropy)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#training_op = optimizer.minimize(loss)
#correct = tf.nn.in_top_k(logits, y, 1)
#accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Multiple layer RNN

#init = tf.global_variables_initializer()

# This should work ..

#with tf.Session() as sess:
#    init.run()
#    for epoch in range(n_epochs):
#        for iteration in range(len(X_train) // batch_size):
#            X_batch, y_batch = next_batch(batch_size, X_train, Y_train)
#            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
#            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
#            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
#    print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
