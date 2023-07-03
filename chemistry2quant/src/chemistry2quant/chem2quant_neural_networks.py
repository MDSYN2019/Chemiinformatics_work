# import the abstract classes
from typing import Any
import os
import numpy as np

# tensorflow libraries
import tensorflow as tf
import tensorflow.keras.layers as layers

# from tensorflow.contrib.layers import fully_connected

# rdkit libraries
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

# deepchem libaries
import deepchem as dc

# Graph convolutional libraries
from deepchem.models.layers import GraphConv, GraphPool, GraphGather

# inhouse classes
from chem2quant_abc import molecular_modelling_neural_network
from chem2quant_pandas import PostgresConnect


class GCNLayer(tf.keras.layers.Layer):
    """
    Implementation of GCN as layer

    https://dmol.pub/dl/gnn.html
    """

    def __init__(self, activation: Any, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        """ """
        node_shape, adj_shape = input_shape
        self.w = self.add_weight(shape=(node_shape[2], node_shape[2]), name="w")

    def call(self, inputs):
        """ """
        nodes, adj = inputs
        degree = tf.reduce_sum(adj, axis=-1)

        # GCN equation
        new_nodes = tf.einsum("bi,bij,bjk,kl->bil", 1 / degree, adj, nodes, self.w)
        out = self.activation(new_nodes)
        return out, adj


"""
Computing losses

In the above models, the loss was computed directly from the model's output. Often that is fine, but not always. 
Consider a classification model that outputs a probability distribution. While it is possible to compute loss from 
the probabilities, it is more numerically stable to compute it from the logs 

Deepchem notes:

Working with Datasets

Data is central to machine learning. It provides a simple but powerful tools for efficiently working with large amounts of data.

DiskDataset, NumpyDataset, ImageDataset 

Every dataset stores a list of samples. Very roughly speaking, a sample is a single data point. In this case, each sample is a molecule. 
In other datasets a sample might correspond to an experimental assay, a cell line, an image, or many other things. For every sample the 
dataset stores the following information:

X: features
y: labels
w: weights
ID: which is a unique identifier for the sample. This can be anything as long as it is unique. Sometimes it is just an integer index. but in this 
    dataset the ID is a SMILES string describing the molecule  


"""


class ClassificationModel(tf.keras.Model):
    """
    notes here:
    https://github.com/deepchem/deepchem/blob/master/examples/tutorials/Creating_Models_with_TensorFlow_and_PyTorch.ipynb
    """

    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(1000, activation="relu")
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        y = self.dense1(inputs)
        if training:
            y = tf.nn.dropout(y, 0.5)  # ..


class NNParameters(molecular_modelling_neural_network):
    """
    Taking the molecular_modelling_neural_network class as the template class

    Not sure I like the multiple inheritance model here but whatever.
    """

    def __init__(
        self,
        layer_points: int,
        n_hidden_layers: int,
        learning_rate: float,
        n_outputs: int,
        data_directory: str,
    ):

        self.layer_points = layer_points
        self.n_hidden_layers = n_hidden_layers
        self.learning_rate = learning_rate
        self.n_outputs = n_outputs
        self.data_directory = data_directory

    def next_batch(self, number_sample: int, data: list, labels: list[str]):
        """
        Return a total of `num` random samples and labels.
        shuffle data
        """
        initial_data = self.connection.return_pandas_table()
        idx = np.arange(0, len(initial_data))  # total amount of data we have
        np.random.shuffle(idx)  # shuffle the data
        idx = idx[:number_sample]  # get a num amount of samples
        data_shuffle = [initial_data[i] for i in idx]  # get the shuffled data
        labels_shuffle = [
            labels[i] for i in idx
        ]  # get the labels for the shuffled data
        self.data_and_label = (np.asarray(data_shuffle), np.asarray(labels_shuffle))

    def gen_training_data(self) -> Any:
        """
        Connect to the postgres database and produce the training data for inputting into
        the deepchem model.


        """
        connection = PostgresConnect(
            user="postgres",
            password="mypassword",
            host="127.0.0.1",
            port="5432",
            database="emolecules",
        )
        connection.make_connection()
        # create the training data
        self.record = (
            connection.return_pandas_table()
        )  # create pandas tables from the original smiles database

        """
        The training data generated looks as follows:

        
        """
        self.training_data = connection.prepare_deepchem_dataset(
            self.record
        )  # prepare the dataset of the smiles strings.

    def _generate_tox21(self) -> None:
        """ """
        tasks, datasets, transformers = dc.molnet.load_tox21(featurizer="GraphConv")
        train_dataset, valid_dataset, test_dataset = datasets

    def _generate_model(self) -> None:
        """ """
        self._keras_model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.layer_points, activation="relu"),
                tf.keras.layers.Dropout(
                    rate=self.learning_rate
                ),  # need to lookup how to apply dropout
                tf.keras.layers.Dense(self.n_outputs),
            ]
        )

    """
    How does this graph convolutional model work?

    
    """

    def _generate_graph_conv_model(self) -> None:
        """ """
        self.graph_conv_layer = GraphConv(128, activation_fn=tf.nn.tanh)
        self.batch_norm1 = layers.BatchNormalization()
        self.gp1 = GraphPool()

    def _compile_model(self) -> None:
        """
        need a good summary of sparse categorical crossentropy, sgd and accuracy
        """
        # how will we compile the model?
        self._keras_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="sgd",
            metrics=["accuracy"],
        )

    def _fit_model(self) -> None:
        """
        Fit the training data onto the keras model
        """
        assert self.training_data != None, "we need training data to be produced first!"
        self._keras_model.fit(self.training_data)

    def prepare_model(self) -> None:
        """ """
        # perpare the sequential neural network inwnards
        # this needs to be made to be more flexible
        keras_model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(1000, activation="relu"),
                tf.keras.layers.Dropout(rate=0.5),
                tf.keras.layers.Dense(1),
            ]
        )
        # why use l2 loss
        model = dc.models.KerasModel(keras_model, dc.models.losses.L2Loss())
        model.fit(self.training_data[0])


def historical():
    # Set up parameters for the neural network
    # this bit can be ignored by using the keras module

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
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits
        )
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


def mol2arr(mol: str) -> np.ndarray:
    """
    Take molecular string representation and return as array
    Morgan fingerprints are basically a reimplementation of ECFP.
    """
    arr = np.zeros((1,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def allocate_train_test(self) -> None:
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

    # make training data and test data
    train_X = np.array([mol2arr(mol) for mol in train_mol])
    train_y = np.array(
        [cls_dic[mol.GetProp("SOL_classification")] for mol in train_mol]
    )
    test_X = np.array([mol2arr(mol) for mol in test_mol])
    test_y = np.array([cls_dic[mol.GetProp("SOL_classification")] for mol in test_mol])


if __name__ == "__main__":
    # need to write very basic excecutable version of the functions above here
    model = NNParameters(
        layer_points=300,
        n_hidden_layers=3,
        learning_rate=0.01,
        n_outputs=1,
        data_directory=".",
    )
    model.gen_training_data()
    model.prepare_model()
