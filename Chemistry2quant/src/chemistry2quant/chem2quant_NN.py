"""
Tensorflow and deepchem implementation of the neural networks that are required 
"""
from typing import Tuple
import os
import numpy as np

# import seaborn as sns
# import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

# import deepchem as dc
# from deepchem.models.tensorgraph.models.graph_models import GraphConvModel

# rdkit part
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from pydantic import BaseModel

class GCNLayer:
    pass


class NNParameters(BaseModel):  # import from pydantic to typecheck the inputs
    layer_points: int
    n_hidden_layers: int
    learning_rate: float
    n_outputs: int

    def __init__(
        self,
        layer_points: int,
        n_hidden_layers: int,
        learning_rate: float,
        n_outputs: int,
    ):
        # check the input parameters of the object with pydantic to ensure that we have checked properly
        super().__init__(
            layer_points=layer_points,
            n_hidden_layers=n_hidden_layers,
            learning_rate=learning_rate,
            n_outputs=n_outputs,
        )

    def next_batch(self, number_sample: int, data: list, labels: list[str]) -> Tuple[[str]]:
        """
        Return a total of `num` random samples and labels.
        """
        idx = np.arange(0, len(data)) # total amount of data we have 
        np.random.shuffle(idx) # shuffle the data 
        idx = idx[:number_sample] # get a num amount of samples 
        data_shuffle = [data[i] for i in idx] # get the shuffled data 
        labels_shuffle = [labels[i] for i in idx] # get the labels for the shuffled data 
        self.data_and_label = (np.asarray(data_shuffle), np.asarray(labels_shuffle))

    def mol2arr(self, mol: str) -> np.ndarray:
        """
        take molecular string representation and return as array
        """
        arr = np.zeros((1,))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def allocate_train_test() -> None:
        """
        Need better description here
        """
        train_mol = [
            mol
            for mol in Chem.SDMolSupplier(os.path.join(datadir, "solubility.train.sdf"))
            if mol != None
        ]
        test_mol = [
            mol
            for mol in Chem.SDMolSupplier(os.path.join(datadir, "solubility.test.sdf"))
            if mol != None
        ]
        cls_mol = list(set([mol.GetProp("SOL_classification") for mol in train_mol]))
        cls_dic = {}

        for i, cl in enumerate(cls_mol):
            cls_dic[cl] = i
            tf.reset_default_graph()
        # make train X, y and test X, y
        train_X = np.array([mol2arr(mol) for mol in train_mol])
        train_y = np.array(
            [cls_dic[mol.GetProp("SOL_classification")] for mol in train_mol]
        )

        test_X = np.array([mol2arr(mol) for mol in test_mol])
        test_y = np.array(
            [cls_dic[mol.GetProp("SOL_classification")] for mol in test_mol]
        )

    def prepare_model(self) -> None:
        """
        Prepare the keras model 
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape = [28, 28]))
        model.add(tf.keras.layers.Flatten())
        # add the layers 
        for layer in self.n_hidden_layers:
            model.add(tf.keras_layers.Dense(layer, activation = 'relu'))
        
# Set up parameters for the neural network
X = tf.placeholder(tf.float32, shape=(None, np.shape(train_X[0])[0]), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
he_init = tf.contrib.layers.variance_scaling_initializer()

with tf.name_scope("dnn"):
    hidden1 = fully_connected(
        X, n_hidden1, weights_initializer=he_init, scope="hidden1"
    )
    hidden2 = fully_connected(
        hidden1, n_hidden2, weights_initializer=he_init, scope="hidden2"
    )
    logits = fully_connected(
        hidden2,
        n_outputs,
        weights_initializer=he_init,
        scope="outputs",
        activation_fn=None,
    )

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
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

if __name__ == "__main__":
    # need to write very basic excecutable version of the functions above here
    pass
